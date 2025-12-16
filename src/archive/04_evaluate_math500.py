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

# SymPy関連 (正解判定用) - Phase 2と同じ強力なものを使用
from latex2sympy2 import latex2sympy
from sympy import simplify

# ==========================================
# 1. 設定パラメータ
# ==========================================
# モデル
POLICY_MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct"   # 生成用
PRM_MODEL_PATH = "models/delta_prm_1.5b_pre_v1"            # 評価用

# データセット
DATASET_NAME = "HuggingFaceH4/MATH-500" # MATHベンチマークの代表的な500問サブセット

# 実験設定
N_SAMPLES = 16          # Best-of-N (16個生成)
MAX_TOKENS = 2048
TEMPERATURE = 0.7

# PRM設定
PRM_BATCH_SIZE = 8
PRM_MAX_LENGTH = 3072   # 学習時と同じ長さを確保
STEP_MERGE_CHARS = 50   # 学習時と同じマージ基準

# ==========================================
# 2. 数学的正解判定ロジック (Phase 2から移植)
# ==========================================

def extract_answer_content(text):
    """\boxed{} の中身を抽出"""
    if not text: return None
    # 最後のboxedを抽出
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
    """
    予測と正解が数学的に等しいか判定する (SymPy + 数値バックアップ)
    """
    if not pred_str or not gold_str: return False
    pred_str = str(pred_str).strip()
    gold_str = str(gold_str).strip()
    
    if pred_str == gold_str: return True

    try:
        # latex2sympy でパースして比較
        sym_pred = latex2sympy(pred_str)
        sym_gold = latex2sympy(gold_str)
        if simplify(sym_pred - sym_gold) == 0:
            return True
    except Exception:
        # 失敗したら数値比較へ
        return robust_float_check(pred_str, gold_str)

    return False

def reduce_step_count(steps, min_chars=50):
    """学習時と同じロジックでステップ結合"""
    merged = []
    buf = ""
    for s in steps:
        if not buf: buf = s; continue
        if len(s) < min_chars or len(buf) < min_chars: buf += "\n" + s
        else: merged.append(buf); buf = s
    if buf: merged.append(buf)
    return merged

# ==========================================
# 3. 評価クラス
# ==========================================

class Evaluator:
    def __init__(self):
        print(f"Loading {DATASET_NAME}...")
        # MATH-500は 'problem', 'solution', 'answer' カラムを持つ
        self.dataset = load_dataset(DATASET_NAME, split="test")
        print(f"Target problems: {len(self.dataset)}")

    def run_generation(self):
        """vLLMで回答生成 (Policy)"""
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
        
        print("Preparing prompts...")
        for item in self.dataset:
            question = item["problem"]
            
            # MATH-500の正解データ処理
            # 'answer' カラムがあればそれを使う、なければ 'solution' から抽出
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

        # 生成実行
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
            
        # メモリ解放
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        torch.cuda.empty_cache()
        print("Generation finished. Released vLLM memory.")
        
        return results

    def run_scoring(self, generated_results):
        """PRMによるスコアリング"""
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
            path_scores = []
            
            for path in paths:
                # 1. ステップ分割と結合 (学習時と同じ前処理)
                raw_steps = [s.strip() for s in re.split(r'\n\s*\n', path) if s.strip()]
                if not raw_steps: 
                    raw_steps = [s.strip() for s in path.split('\n') if s.strip()]
                
                steps = reduce_step_count(raw_steps, min_chars=STEP_MERGE_CHARS)
                
                if not steps:
                    path_scores.append(-99.0)
                    continue
                
                # 2. 全ステップを評価して最小値(Min)を取る
                # 入力作成: [Problem + Step1], [Problem + Step1 + Step2]...
                step_inputs = []
                curr_text = problem
                for step in steps:
                    curr_text += "\n" + step
                    step_inputs.append(curr_text)
                
                step_rewards = []
                with torch.no_grad():
                    # バッチ推論
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
                        step_rewards.extend(out.logits.squeeze(-1).tolist())
                
                # パススコア = Min(ステップ報酬)
                # どんなに良くても一度でも致命的なミス(低い値)があれば低評価にする
                final_score = min(step_rewards) if step_rewards else -99.0
                path_scores.append(final_score)
            
            item["scores"] = path_scores
            scored_results.append(item)
            
        return scored_results

    def calculate_metrics(self, results):
        """3つの指標を計算して比較"""
        print("Calculating metrics...")
        
        # カウンター
        pass1_total_correct = 0  # 生成された全パスのうち正解だった数 (平均計算用)
        total_generated_paths = 0
        
        maj_correct_count = 0    # 多数決で正解した問題数
        prm_correct_count = 0    # PRMで正解した問題数
        total_problems = len(results)
        
        for item in tqdm(results, desc="Checking Correctness"):
            gold = item["gold"]
            paths = item["paths"]
            scores = item["scores"]
            
            # 各パスから答えを抽出
            extracted_answers = [extract_answer_content(p) for p in paths]
            
            # --- 1. Pass@1 (Average Accuracy) ---
            # 生成されたN個のパスそれぞれの正誤を判定
            path_correctness = []
            valid_answers_for_voting = []
            
            for ans in extracted_answers:
                is_correct = check_correctness(ans, gold)
                path_correctness.append(is_correct)
                if ans: valid_answers_for_voting.append(ans)
            
            pass1_total_correct += sum(path_correctness)
            total_generated_paths += len(paths)
            
            # --- 2. Majority Voting ---
            if valid_answers_for_voting:
                # 単純な文字列一致での多数決 (表記揺れはSymPyで吸収できないため文字列ベースが一般的)
                # ただし、厳密には「正規化後の文字列」で投票するのが良いが、ここでは簡易版
                vote = Counter(valid_answers_for_voting).most_common(1)[0][0]
                if check_correctness(vote, gold):
                    maj_correct_count += 1
            
            # --- 3. Delta-PRM (Best-of-N) ---
            # スコアが最大のパスを選択
            best_idx = np.argmax(scores)
            best_ans = extracted_answers[best_idx]
            
            if check_correctness(best_ans, gold):
                prm_correct_count += 1

        # 結果集計
        pass1_acc = pass1_total_correct / total_generated_paths
        maj_acc = maj_correct_count / total_problems
        prm_acc = prm_correct_count / total_problems
        
        print("\n" + "="*40)
        print(f"EVALUATION RESULTS on {DATASET_NAME} (N={N_SAMPLES})")
        print("="*40)
        print(f"1. Pass@1 (Avg) : {pass1_acc:.2%} (Model's raw capability)")
        print(f"2. Majority Vote: {maj_acc:.2%} (Consensus baseline)")
        print(f"3. Delta-PRM    : {prm_acc:.2%} (Ours)")
        print("="*40)
        
        # 勝利判定
        if prm_acc > maj_acc:
            print("🏆 Delta-PRM outperforms Majority Voting!")
        elif prm_acc > pass1_acc:
            print("✅ Delta-PRM improves over Pass@1 (but lost to Voting)")
        else:
            print("⚠️ Delta-PRM needs improvement.")

def main():
    evaluator = Evaluator()
    
    # 1. 生成
    results = evaluator.run_generation()
    
    # 2. 採点
    scored_results = evaluator.run_scoring(results)
    
    # 3. 評価
    evaluator.calculate_metrics(scored_results)
    
    # 結果保存
    with open("data/math500_results.json", "w") as f:
        json.dump(scored_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()