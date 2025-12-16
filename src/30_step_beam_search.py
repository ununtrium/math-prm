import os
import torch
import json
import re
import argparse
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed

# ==========================================
# ★ vLLMのインポート
# ==========================================
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vllm not found. Please install via: pip install vllm")
    exit()

# 数式処理
try:
    from latex2sympy2 import latex2sympy
    from sympy import simplify
except ImportError:
    print("Warning: latex2sympy2 or sympy not found.")

# ==========================================
# 設定
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--prm_model", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="data/math500_results_vllm.json")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--num_candidates", type=int, default=5)
    
    # vLLM用のGPUメモリ設定
    # エラーが出る場合は 0.5 や 0.4 に下げてPRM用の領域を確保してください
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    
    # コンテキスト長の上限 (エラー回避用)
    parser.add_argument("--max_model_len", type=int, default=4096)

    return parser.parse_args()

# ==========================================
# ユーティリティ (正解判定)
# ==========================================
def extract_answer(text):
    if not text: return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches: return matches[-1]
    return None

def robust_float_check(pred, gold):
    try:
        def clean_to_float(s):
            s = str(s).replace(r"\frac", "").replace("{", "(").replace("}", ")").replace("^", "**")
            s = s.replace(r"\left", "").replace(r"\right", "").replace(",", "").replace("$", "")
            return float(eval(s))
        if not any(c.isalpha() for c in str(pred)) and not any(c.isalpha() for c in str(gold)):
            return abs(clean_to_float(pred) - clean_to_float(gold)) < 1e-6
    except: pass
    return False

def check_correctness(pred_str, gold_str):
    if not pred_str or not gold_str: return False
    pred_str = str(pred_str).strip(); gold_str = str(gold_str).strip()
    if pred_str == gold_str: return True
    try:
        sym_pred = latex2sympy(pred_str); sym_gold = latex2sympy(gold_str)
        if simplify(sym_pred - sym_gold) == 0: return True
    except: pass
    return robust_float_check(pred_str, gold_str)

# ==========================================
# vLLM対応 ビームサーチ・クラス
# ==========================================
class VLLMStepBeamSearcher:
    def __init__(self, llm_engine, prm_model, prm_tokenizer, device):
        self.llm = llm_engine
        self.prm_model = prm_model
        self.prm_tokenizer = prm_tokenizer
        self.device = device

    def get_prm_scores_batch(self, texts, batch_size=8):
        scores = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # パディングありでバッチ処理
            inputs = self.prm_tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=3072
            ).to(self.device)
            
            with torch.no_grad():
                out = self.prm_model(**inputs)
            scores.extend(out.logits.squeeze(-1).tolist())
        return scores

    def search(self, problem, beam_width=5, max_steps=20, num_candidates=5):
        beam = [(problem, 0.0, [])]
        
        # 安全マージン (max_model_len より少し小さく設定)
        MAX_TOKEN_LIMIT = 3500 
        
        sampling_params = SamplingParams(
            n=num_candidates,
            temperature=0.7,
            top_p=0.95,
            max_tokens=512,
            stop=["\n\n", "The final answer", "Wait"]
        )

        for t in range(max_steps):
            prompts = []
            metadata = [] 
            finished_paths = []
            
            # --- 1. Expand & ガード処理 ---
            for idx, (curr_text, curr_score, history) in enumerate(beam):
                if extract_answer(curr_text):
                    finished_paths.append((curr_text, curr_score, history))
                    continue
                
                # 長さガード (文字数ベースの簡易版)
                if len(curr_text) > MAX_TOKEN_LIMIT * 3.5:
                    finished_paths.append((curr_text, curr_score, history))
                    continue

                prompts.append(curr_text)
                metadata.append((curr_text, history))
            
            if not prompts:
                if finished_paths:
                    finished_paths.sort(key=lambda x: x[1], reverse=True)
                    return finished_paths[0]
                # 全滅時
                return beam[0]

            # --- 2. vLLMで一括生成 ---
            try:
                outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            except ValueError as e:
                print(f"Warning: vLLM generation failed (length limit?). Dropping batch. Error: {e}")
                if finished_paths:
                    finished_paths.sort(key=lambda x: x[1], reverse=True)
                    return finished_paths[0]
                return beam[0]

            # --- 3. PRM評価用のリスト作成 ---
            next_candidates_text = []
            next_candidates_info = [] 
            
            for i, request_output in enumerate(outputs):
                curr_text, history = metadata[i]
                for completion in request_output.outputs:
                    step = completion.text
                    # vLLMはstop tokenを除去して返すので、ここでのsplitは念の為
                    if "\n\n" in step: step = step.split("\n\n")[0]
                    step = step.strip()
                    
                    if not step: continue
                    
                    # PRM入力 (\n結合)
                    new_text = curr_text + "\n" + step
                    next_candidates_text.append(new_text)
                    next_candidates_info.append((new_text, step, history))

            if not next_candidates_text:
                if finished_paths:
                    finished_paths.sort(key=lambda x: x[1], reverse=True)
                    return finished_paths[0]
                break

            # --- 4. PRM一括採点 ---
            scores = self.get_prm_scores_batch(next_candidates_text, batch_size=8)
            
            # --- 5. Select (選抜) ---
            candidates_pool = []
            for i, score in enumerate(scores):
                new_text, step, hist = next_candidates_info[i]
                candidates_pool.append((new_text, score, hist + [step]))
            
            candidates_pool.extend(finished_paths)
            candidates_pool.sort(key=lambda x: x[1], reverse=True)
            beam = candidates_pool[:beam_width]
            
            if all(extract_answer(p[0]) for p in beam):
                break
        
        return beam[0]

# ==========================================
# メイン処理
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)
    
    # GPUデバイスの自動選択 (PRM用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. PRM Load
    print(f"Loading PRM: {args.prm_model}")
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_model)
    
    # ★重要修正: Pad Tokenの設定漏れ対策
    if prm_tokenizer.pad_token is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token
        
    prm_model = AutoModelForSequenceClassification.from_pretrained(
        args.prm_model, 
        num_labels=1, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    prm_model.eval()
    
    # 2. Generator (vLLM) Load
    print(f"Loading vLLM Generator: {args.gen_model} (Max Len: {args.max_model_len})")
    llm = LLM(
        model=args.gen_model, 
        tensor_parallel_size=1, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
        # ★重要修正: これがないと長文で落ちる
        max_model_len=args.max_model_len 
    )

    searcher = VLLMStepBeamSearcher(llm, prm_model, prm_tokenizer, device)

    # Dataset Load
    print("Loading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/math-500", split="test")

    results = []
    correct_count = 0
    
    print("\nStarting vLLM Evaluation...")
    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        problem = item["problem"]
        gold_answer = extract_answer(item["solution"])
        
        best_path_text, best_score, steps = searcher.search(
            problem, 
            beam_width=args.beam_width, 
            max_steps=args.max_steps,
            num_candidates=args.num_candidates
        )
        
        pred_answer = extract_answer(best_path_text)
        is_correct = check_correctness(pred_answer, gold_answer)
        
        if is_correct: correct_count += 1
        results.append({
            "problem": problem, "gold": gold_answer, "pred": pred_answer,
            "is_correct": is_correct, "score": best_score, "steps": steps
        })
        
        if (i+1) % 10 == 0:
            print(f"Done {i+1}/{len(dataset)} | Acc: {correct_count/(i+1):.2%}")

    acc = correct_count / len(dataset)
    print(f"Final Accuracy: {acc:.2%}")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "details": results}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()