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
import gc

# SymPy関連
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 1. 設定パラメータ (デフォルト値)
# ==========================================
POLICY_MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
DATASET_NAME = "HuggingFaceH4/MATH-500"

# 推論設定
N_SAMPLES = 16
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# PRM設定
PRM_BATCH_SIZE = 8
PRM_MAX_LENGTH = 3072
STEP_MERGE_CHARS = 50
TARGET_MAX_STEPS = 15

# ==========================================
# 2. ユーティリティ関数
# ==========================================
def extract_answer_content(text):
    if not text: return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches: return matches[-1].strip()
    return None

def robust_float_check(pred, gold):
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
    if len(steps) <= target_max: return steps
    merged_steps = []
    buffer = ""
    for i, step in enumerate(steps):
        if i == 0: buffer = step; continue
        if len(step) < min_chars or len(buffer) < min_chars:
            buffer += "\n" + step
        else:
            merged_steps.append(buffer); buffer = step
    if buffer: merged_steps.append(buffer)
    while len(merged_steps) > target_max:
        new_merged = []
        for i in range(0, len(merged_steps), 2):
            if i + 1 < len(merged_steps): new_merged.append(merged_steps[i] + "\n" + merged_steps[i+1])
            else: new_merged.append(merged_steps[i])
        merged_steps = new_merged
        if len(merged_steps) <= 1: break
    return merged_steps

# ==========================================
# 3. 評価クラス (修正版)
# ==========================================
class Evaluator:
    def __init__(self):
        print(f"Loading {DATASET_NAME}...")
        self.dataset = load_dataset(DATASET_NAME, split="test")
        print(f"Target problems: {len(self.dataset)}")

    def run_generation(self, seed=42):
        """Seedを指定して生成"""
        print(f"Initializing Policy Model (Seed={seed})...")
        llm = LLM(
            model=POLICY_MODEL_ID,
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            seed=seed
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

            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            raw_data.append({"question": question, "gold": gold})

        print(f"Generating {N_SAMPLES} paths per problem...")
        params = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, seed=seed)
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

    def run_scoring(self, generated_results, model_path):
        """モデルパスを指定して採点 (修正版: ステップテキストも保存)"""
        print(f"Initializing PRM/ORM Model ({model_path})...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        
        scored_results = []
        for item in tqdm(generated_results, desc="Scoring"):
            problem = item["problem"]
            paths = item["paths"]
            
            path_scores = []
            path_step_scores = []
            path_steps_text = [] # ★追加: 分割後のステップテキスト
            
            for path in paths:
                raw_steps = [s.strip() for s in re.split(r'\n\s*\n', path) if s.strip()]
                if not raw_steps: raw_steps = [s.strip() for s in path.split('\n') if s.strip()]
                
                # 学習時と同じロジックで結合
                steps = reduce_step_count(raw_steps, target_max=TARGET_MAX_STEPS, min_chars=STEP_MERGE_CHARS)
                
                if not steps:
                    path_scores.append(-99.0)
                    path_step_scores.append([-99.0])
                    path_steps_text.append([]) # 空リスト
                    continue
                
                # 入力作成
                step_inputs = []
                curr_text = problem
                for step in steps:
                    curr_text += "\n" + step
                    step_inputs.append(curr_text)
                
                # 推論
                current_rewards = []
                with torch.no_grad():
                    for i in range(0, len(step_inputs), PRM_BATCH_SIZE):
                        batch = step_inputs[i : i+PRM_BATCH_SIZE]
                        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=PRM_MAX_LENGTH).to(model.device)
                        out = model(**inputs)
                        current_rewards.extend(out.logits.squeeze(-1).tolist())
                
                final_score = min(current_rewards) if current_rewards else -99.0
                
                path_scores.append(final_score)
                path_step_scores.append(current_rewards)
                path_steps_text.append(steps) # ★保存
            
            new_item = item.copy()
            new_item["scores"] = path_scores
            new_item["step_scores"] = path_step_scores
            new_item["step_texts"] = path_steps_text # ★結果に追加
            
            scored_results.append(new_item)
            
        del model
        torch.cuda.empty_cache()
        return scored_results

    def calculate_metrics(self, results, scale=0.5):
        """数値を辞書で返すように変更"""
        metrics = {
            "pass1": 0, "maj_vote": 0,
            "bon_min": 0, "bon_last": 0, "bon_mean": 0,
            "weighted_vote": 0
        }
        total = len(results)
        total_paths = 0
        
        for item in results:
            gold = item["gold"]
            paths = item["paths"]
            step_scores_list = item["step_scores"]
            extracted = [extract_answer_content(p) for p in paths]
            valid_idx = [i for i, a in enumerate(extracted) if a]
            
            # Pass@1
            for i, ans in enumerate(extracted):
                if check_correctness(ans, gold): metrics["pass1"] += 1
            total_paths += len(paths)
            
            if not valid_idx: continue

            # Majority Vote
            valid_ans = [extracted[i] for i in valid_idx]
            maj = Counter(valid_ans).most_common(1)[0][0]
            if check_correctness(maj, gold): metrics["maj_vote"] += 1
            
            # BoN (Min)
            scores_min = [min(s) if s else -99 for s in step_scores_list]
            best_min = extracted[np.argmax(scores_min)]
            if check_correctness(best_min, gold): metrics["bon_min"] += 1

            # BoN (Last)
            scores_last = [s[-1] if s else -99 for s in step_scores_list]
            best_last = extracted[np.argmax(scores_last)]
            if check_correctness(best_last, gold): metrics["bon_last"] += 1
            
            # BoN (Mean)
            scores_mean = [np.mean(s) if s else -99 for s in step_scores_list]
            best_mean = extracted[np.argmax(scores_mean)]
            if check_correctness(best_mean, gold): metrics["bon_mean"] += 1

            # Weighted Vote (Mean base, Scale=0.5)
            votes = {}
            for i in valid_idx:
                ans = extracted[i]
                sc = scores_mean[i] # Meanスコアを使用
                w = np.exp(sc * scale)
                votes[ans] = votes.get(ans, 0) + w
            if check_correctness(max(votes, key=votes.get), gold): metrics["weighted_vote"] += 1

        return {
            "pass1": metrics["pass1"]/total_paths,
            "maj_vote": metrics["maj_vote"]/total,
            "bon_min": metrics["bon_min"]/total,
            "bon_last": metrics["bon_last"]/total,
            "bon_mean": metrics["bon_mean"]/total,
            "weighted_vote": metrics["weighted_vote"]/total
        }

if __name__ == "__main__":
    # テスト実行用 (本番はsrc/12から呼ぶ)
    print("Please run src/12_run_multi_trials_all.py for the full experiment.")