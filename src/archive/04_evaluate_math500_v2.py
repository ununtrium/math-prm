import json
import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from collections import Counter

# SymPy関連
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 1. 設定パラメータ
# ==========================================
POLICY_MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
PRM_MODEL_PATH = "models/orm_1.5b_30k_v1.0"
DATASET_NAME = "HuggingFaceH4/MATH-500"

# 推論設定
N_SAMPLES = 16          # Best-of-N
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# PRM設定
PRM_BATCH_SIZE = 8
PRM_MAX_LENGTH = 3072   # 学習時(3072)に合わせる
STEP_MERGE_CHARS = 50   # 学習時のマージ基準
TARGET_MAX_STEPS = 15   # 学習時の最大ステップ数

# ==========================================
# 2. ユーティリティ関数
# ==========================================

def extract_answer_content(text):
    """\boxed{} の中身を抽出"""
    if not text: return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches: return matches[-1].strip()
    return None

def robust_float_check(pred, gold):
    """数値的なバックアップ判定"""
    try:
        def to_float(s):
            s = str(s).replace(r"\frac", "").replace("{", "(").replace("}", ")").replace("^", "**")
            s = s.replace(r"\left", "").replace(r"\right", "").replace(",", "")
            return float(eval(s))
        if not any(c.isalpha() for c in str(pred)) and not any(c.isalpha() for c in str(gold)):
            return abs(to_float(pred) - to_float(gold)) < 1e-6
    except:
        pass
    return False

def check_correctness(pred_str, gold_str):
    """正解判定 (SymPy + 数値)"""
    if not pred_str or not gold_str: return False
    pred_str = str(pred_str).strip()
    gold_str = str(gold_str).strip()
    if pred_str == gold_str: return True
    try:
        sym_pred = latex2sympy(pred_str)
        sym_gold = latex2sympy(gold_str)
        if simplify(sym_pred - sym_gold) == 0:
            return True
    except Exception:
        return robust_float_check(pred_str, gold_str)
    return False

def reduce_step_count(steps, target_max=15, min_chars=50):
    """
    [完全版] 学習時と同じロジックでステップ数を調整する。
    1. 短い行の結合 (Phase 1)
    2. 上限を超える場合の強制ペアリング (Phase 2)
    """
    if len(steps) <= target_max: return steps

    # Phase 1: Semantic Merge
    merged_steps = []
    buffer = ""
    for i, step in enumerate(steps):
        if i == 0: buffer = step; continue
        if len(step) < min_chars or len(buffer) < min_chars:
            buffer += "\n" + step
        else:
            merged_steps.append(buffer); buffer = step
    if buffer: merged_steps.append(buffer)

    # Phase 2: Forced Merge
    while len(merged_steps) > target_max:
        new_merged = []
        for i in range(0, len(merged_steps), 2):
            if i + 1 < len(merged_steps):
                new_merged.append(merged_steps[i] + "\n" + merged_steps[i+1])
            else:
                new_merged.append(merged_steps[i])
        merged_steps = new_merged
        if len(merged_steps) <= 1: break
    return merged_steps

# ==========================================
# 3. 評価クラス
# ==========================================

class Evaluator:
    def __init__(self):
        print(f"Loading {DATASET_NAME}...")
        self.dataset = load_dataset(DATASET_NAME, split="test")
        print(f"Target problems: {len(self.dataset)}")

    def run_generation(self):
        print(f"Initializing Policy Model ({POLICY_MODEL_ID})...")
        llm = LLM(
            model=POLICY_MODEL_ID,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            dtype="bfloat16"
        )
        tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_ID)
        
        prompts = []
        raw_data = []
        system_prompt = "Please reason step by step and put your final answer within \\boxed{}."
        
        for item in self.dataset:
            question = item["problem"]
            if "answer" in item and item["answer"]:
                gold = item["answer"]
            else:
                gold = extract_answer_content(item["solution"])

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            raw_data.append({"question": question, "gold": gold})

        print(f"Generating {N_SAMPLES} paths per problem...")
        params = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        outputs = llm.generate(prompts, params)
        
        results = []
        for i, output in enumerate(outputs):
            paths = [o.text for o in output.outputs]
            results.append({
                "problem": raw_data[i]["question"],
                "gold": raw_data[i]["gold"],
                "paths": paths
            })
            
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        torch.cuda.empty_cache()
        return results

    def run_scoring(self, generated_results):
        print(f"Initializing PRM Model ({PRM_MODEL_PATH})...")
        prm_tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL_PATH)
        prm_model = AutoModelForSequenceClassification.from_pretrained(
            PRM_MODEL_PATH, 
            num_labels=1, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        prm_model.eval()
        
        scored_results = []
        
        print("Scoring paths...")
        for item in tqdm(generated_results, desc="PRM Scoring"):
            problem = item["problem"]
            paths = item["paths"]
            
            # 各パスの「集計スコア(min)」と「全ステップスコア(list)」を保存
            path_scores = []
            path_step_scores = []
            
            for path in paths:
                # 1. ステップ分割 (学習時と同じロジック)
                raw_steps = [s.strip() for s in re.split(r'\n\s*\n', path) if s.strip()]
                if not raw_steps: 
                    raw_steps = [s.strip() for s in path.split('\n') if s.strip()]
                
                # 2. ステップ結合・圧縮 (学習時と同じロジック)
                # ★修正: target_maxを指定して強制圧縮
                steps = reduce_step_count(raw_steps, target_max=TARGET_MAX_STEPS, min_chars=STEP_MERGE_CHARS)
                
                if not steps:
                    path_scores.append(-99.0)
                    path_step_scores.append([-99.0])
                    continue
                
                step_inputs = []
                curr_text = problem
                for step in steps:
                    curr_text += "\n" + step
                    step_inputs.append(curr_text)
                
                # 3. 推論
                current_rewards = []
                with torch.no_grad():
                    for i in range(0, len(step_inputs), PRM_BATCH_SIZE):
                        batch = step_inputs[i : i+PRM_BATCH_SIZE]
                        inputs = prm_tokenizer(
                            batch, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=PRM_MAX_LENGTH
                        ).to(prm_model.device)
                        
                        out = prm_model(**inputs)
                        current_rewards.extend(out.logits.squeeze(-1).tolist())
                
                # 4. 集計と保存
                final_score = min(current_rewards) if current_rewards else -99.0
                
                path_scores.append(final_score)
                path_step_scores.append(current_rewards)
            
            item["scores"] = path_scores
            item["step_scores"] = path_step_scores # 生スコアも保存
            scored_results.append(item)
            
        return scored_results

def main():
    evaluator = Evaluator()
    
    # 1. 生成
    results = evaluator.run_generation()
    
    # 2. 採点
    scored_results = evaluator.run_scoring(results)
    
    # 3. 保存 (後で src/05_recalculate.py 等で分析可能)
    output_file = "data/math500_results_full_scores_orm_30k_v1.0.json"
    print(f"Saving full results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(scored_results, f, ensure_ascii=False, indent=2)
        
    print("Done! Evaluation finished.")

if __name__ == "__main__":
    main()