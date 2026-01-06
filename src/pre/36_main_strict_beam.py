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

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vllm not found. Please install via: pip install vllm")
    exit()

try:
    from latex2sympy2 import latex2sympy
    from sympy import simplify
except ImportError:
    print("Warning: latex2sympy2 or sympy not found.")

# ==========================================
# 1. 設定 & ユーティリティ
# ==========================================
# アノテーション時の設定と一致させる
MIN_STEP_CHARS = 50 
MAX_STEP_COUNT = 15

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

def clean_text_for_prm(text):
    # PRM学習時のフォーマットに合わせてタグを削除
    text = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>user\s*", "", text)
    text = re.sub(r"<\|im_end\|>\s*<\|im_start\|>assistant\s*", "\n", text)
    text = re.sub(r"<\|im_end\|>\s*$", "", text)
    text = text.replace("<|im_start|>assistant", "").replace("<|im_end|>", "")
    return text.strip()

"""
# ==========================================
# 2. Strict Adaptive Beam Search Class
# ==========================================
class StrictAdaptiveBeamSearcher:
    def __init__(self, llm_engine, prm_model, prm_tokenizer, device, seed):
        self.llm = llm_engine
        self.prm_model = prm_model
        self.prm_tokenizer = prm_tokenizer
        self.device = device
        self.seed = seed

    def get_prm_scores_batch(self, texts, batch_size=8):
        scores = []
        # メモリ節約のため勾配計算なし
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                # PRMに入力する前に整形が必要ならここで行う
                clean_texts = [clean_text_for_prm(t) for t in batch_texts]
                
                inputs = self.prm_tokenizer(
                    clean_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
                ).to(self.device)
                out = self.prm_model(**inputs)
                scores.extend(out.logits.squeeze(-1).tolist())
        return scores

    def search(self, problem_text, beam_width=5, max_gen_loops=30, num_candidates=3):
        
        #max_gen_loops: 生成ループの最大回数（マージが発生するため、ステップ数より多く設定する）
        
        # Beam State: (full_text, min_score, current_buffer, step_count, history_log)
        # current_buffer: 直前の \n\n 以降の、まだ評価条件(50文字)を満たしていないテキスト
        beam = [(problem_text, 0.0, "", 0, [])]
        
        all_finished_paths = []
        
        # Stopトークン設定: 段落区切り "\n\n" で止める
        sampling_params = SamplingParams(
            n=num_candidates, temperature=0.7, top_p=0.95, max_tokens=512,
            stop=["\n\n", "The final answer", "<|im_end|>"], 
            seed=self.seed
        )

        for loop_idx in range(max_gen_loops):
            prompts = []
            metadata = [] # (parent_idx)

            # 1. Expand
            for i, (txt, min_scr, buf, cnt, hist) in enumerate(beam):
                if extract_answer(txt):
                    all_finished_paths.append((txt, min_scr, hist, "finished"))
                    continue
                
                # トークン長制限 or ステップ数上限
                if len(txt) > 12000 or cnt >= MAX_STEP_COUNT:
                    all_finished_paths.append((txt, min_scr, hist, "max_limit"))
                    continue

                prompts.append(txt)
                metadata.append(i)

            if not prompts: break

            # 2. Generate
            try:
                outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            except ValueError: break

            # 3. Process Logic (Strict Merge)
            next_candidates = [] 
            eval_queue_texts = [] 
            eval_queue_indices = []

            for i, request_output in enumerate(outputs):
                parent_idx = metadata[i]
                parent_text, parent_min, parent_buffer, parent_count, parent_hist = beam[parent_idx]

                for completion in request_output.outputs:
                    step_fragment = completion.text.replace("<|im_end|>", "").strip()
                    
                    # 生成が空でバッファもない場合はスキップ
                    if not step_fragment and not parent_buffer: continue

                    # テキスト結合ロジック
                    # Generatorは \n\n の手前で止まっているので、繋ぐときは \n\n を補うのが自然
                    # ただし初回や文脈によっては \n だけの場合もあるが、
                    # ここでは「段落生成」として統一的に \n\n を入れる
                    
                    connector = "\n\n"
                    # バッファがある(=前回マージされた)場合、テキスト上は既に繋がっているので
                    # バッファの中身として蓄積するイメージ
                    
                    new_full_text = parent_text + connector + step_fragment
                    
                    # バッファの更新 (評価用テキスト長チェックのため)
                    # 前回のバッファ + 改行 + 今回の断片
                    separator = "\n" if parent_buffer else ""
                    new_buffer = parent_buffer + separator + step_fragment
                    
                    has_answer = extract_answer(new_full_text)
                    
                    # ★厳密な評価トリガー★
                    should_evaluate = False
                    if has_answer:
                        should_evaluate = True
                    elif len(new_buffer) >= MIN_STEP_CHARS:
                        should_evaluate = True
                    else:
                        should_evaluate = False

                    if should_evaluate:
                        # PRM評価リストに追加
                        eval_queue_texts.append(new_full_text)
                        
                        next_candidates.append({
                            "text": new_full_text,
                            "min_score": parent_min, # 更新待ち
                            "buffer": "",            # クリア
                            "count": parent_count + 1,
                            "history": parent_hist,  # 更新待ち
                            "needs_eval": True
                        })
                        eval_queue_indices.append(len(next_candidates) - 1)
                    else:
                        # 評価スキップ（マージ）
                        next_candidates.append({
                            "text": new_full_text,
                            "min_score": parent_min, # 親のスコア維持
                            "buffer": new_buffer,    # バッファ蓄積
                            "count": parent_count,   # ステップ数増えない
                            "history": parent_hist,
                            "needs_eval": False
                        })

            # 4. Batch Scoring
            if eval_queue_texts:
                scores = self.get_prm_scores_batch(eval_queue_texts)
                for idx_in_queue, score in enumerate(scores):
                    candidate_idx = eval_queue_indices[idx_in_queue]
                    cand = next_candidates[candidate_idx]
                    
                    # Minスコア更新
                    if cand["count"] == 1:
                        cand["min_score"] = score
                    else:
                        cand["min_score"] = min(cand["min_score"], score)
                    
                    # 履歴保存
                    new_hist_entry = {
                        "step_idx": cand["count"],
                        "score": score,
                        "min_score": cand["min_score"]
                    }
                    cand["history"] = cand["history"] + [new_hist_entry]

            # 5. Select (Pruning)
            pool = []
            for c in next_candidates:
                pool.append((c["text"], c["min_score"], c["buffer"], c["count"], c["history"]))
                
                # 答えが出ているものは Finished プールへもコピー
                if extract_answer(c["text"]):
                    all_finished_paths.append((c["text"], c["min_score"], c["history"], "finished"))

            # Minスコア順にソート (保留中のパスも親スコアで競争に参加)
            pool.sort(key=lambda x: x[1], reverse=True)
            beam = pool[:beam_width]

            # 全員が答えに到達していれば終了
            if beam and all(extract_answer(b[0]) for b in beam):
                break

        # 最終選択ロジック
        final_selection = None
        status = "failed"

        # 1. 完了したパスの中からベストを選ぶ
        if all_finished_paths:
            # 重複排除
            unique = {}
            for p in all_finished_paths:
                unique[p[0]] = p
            candidates = list(unique.values())
            candidates.sort(key=lambda x: x[1], reverse=True)
            final_selection = candidates[0]
            status = "success"
        
        # 2. 完了パスがない場合、ビームに残ったベスト(未完)を選ぶ
        elif beam:
            beam.sort(key=lambda x: x[1], reverse=True)
            final_selection = beam[0][:4] # full_text, min, hist, status用のダミー
            status = "timeout_no_answer"
        
        else:
            return None, -999, [], "failed"

        return final_selection[0], final_selection[1], final_selection[2], status
"""

# ==========================================
# 修正版: Strict Adaptive Beam Search (v2)
# ==========================================
class StrictAdaptiveBeamSearcher:
    def __init__(self, llm_engine, prm_model, prm_tokenizer, device, seed):
        self.llm = llm_engine
        self.prm_model = prm_model
        self.prm_tokenizer = prm_tokenizer
        self.device = device
        self.seed = seed

    def get_prm_scores_batch(self, texts, batch_size=8):
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                # PRM学習時と同じクリーニング
                clean_texts = [clean_text_for_prm(t) for t in batch_texts]
                inputs = self.prm_tokenizer(
                    clean_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048
                ).to(self.device)
                out = self.prm_model(**inputs)
                scores.extend(out.logits.squeeze(-1).tolist())
        return scores

    def search(self, problem_text, beam_width=5, max_gen_loops=40, num_candidates=3):
        # max_gen_loops を少し増やしました（\nで細かく止まるため回数が必要）
        
        # Beam: (full_text, min_score, current_buffer, step_count, history)
        beam = [(problem_text, 0.0, "", 0, [])]
        all_finished_paths = []
        
        # ★修正1: "\n" も停止条件に含める（Qwenの挙動に合わせる）
        sampling_params = SamplingParams(
            n=num_candidates, temperature=0.7, top_p=0.95, max_tokens=512,
            stop=["\n\n", "\n", "The final answer", "<|im_end|>"], 
            seed=self.seed
        )

        for loop_idx in range(max_gen_loops):
            prompts = []
            metadata = [] 

            # 1. Expand
            for i, (txt, min_scr, buf, cnt, hist) in enumerate(beam):
                if extract_answer(txt):
                    all_finished_paths.append((txt, min_scr, hist, "finished"))
                    continue
                # 長すぎる、またはステップ数オーバー
                if len(txt) > 12000 or cnt >= MAX_STEP_COUNT:
                    all_finished_paths.append((txt, min_scr, hist, "max_limit"))
                    continue

                prompts.append(txt)
                metadata.append(i)

            if not prompts: break

            # 2. Generate
            try:
                outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            except ValueError: break

            # 3. Process Logic
            next_candidates = [] 
            eval_queue_texts = [] 
            eval_queue_indices = []

            for i, request_output in enumerate(outputs):
                parent_idx = metadata[i]
                parent_text, parent_min, parent_buffer, parent_count, parent_hist = beam[parent_idx]

                for completion in request_output.outputs:
                    step_fragment = completion.text.replace("<|im_end|>", "").strip()
                    
                    # 何も生成されなかった場合
                    if not step_fragment and not parent_buffer: continue

                    # ★修正2: コネクタを "\n" に統一
                    # 前回のバッファがある場合、バッファ内での結合用セパレータも "\n"
                    # これにより PRM は常に "Line1 \n Line2 \n Line3" という自然な形式を見れる
                    
                    connector = "\n"
                    
                    # 文脈としての全文更新
                    new_full_text = parent_text + connector + step_fragment
                    
                    # バッファ更新 (評価判定用)
                    buffer_separator = "\n" if parent_buffer else ""
                    new_buffer = parent_buffer + buffer_separator + step_fragment
                    
                    has_answer = extract_answer(new_full_text)
                    
                    # 評価判定ロジック (ここは同じ)
                    should_evaluate = False
                    if has_answer:
                        should_evaluate = True
                    elif len(new_buffer) >= MIN_STEP_CHARS:
                        should_evaluate = True
                    else:
                        should_evaluate = False

                    if should_evaluate:
                        eval_queue_texts.append(new_full_text)
                        next_candidates.append({
                            "text": new_full_text,
                            "min_score": parent_min, 
                            "buffer": "",            
                            "count": parent_count + 1,
                            "history": parent_hist,
                            "needs_eval": True
                        })
                        eval_queue_indices.append(len(next_candidates) - 1)
                    else:
                        # 評価スキップ（バッファリング継続）
                        next_candidates.append({
                            "text": new_full_text,
                            "min_score": parent_min, 
                            "buffer": new_buffer,    
                            "count": parent_count,   
                            "history": parent_hist,
                            "needs_eval": False
                        })

            # 4. Batch Scoring
            if eval_queue_texts:
                scores = self.get_prm_scores_batch(eval_queue_texts)
                for idx_in_queue, score in enumerate(scores):
                    candidate_idx = eval_queue_indices[idx_in_queue]
                    cand = next_candidates[candidate_idx]
                    
                    if cand["count"] == 1:
                        cand["min_score"] = score
                    else:
                        cand["min_score"] = min(cand["min_score"], score)
                    
                    new_hist_entry = {
                        "step_idx": cand["count"],
                        "score": score,
                        "min_score": cand["min_score"]
                    }
                    cand["history"] = cand["history"] + [new_hist_entry]

            # 5. Select
            pool = []
            for c in next_candidates:
                pool.append((c["text"], c["min_score"], c["buffer"], c["count"], c["history"]))
                if extract_answer(c["text"]):
                    all_finished_paths.append((c["text"], c["min_score"], c["history"], "finished"))

            pool.sort(key=lambda x: x[1], reverse=True)
            beam = pool[:beam_width]

            if beam and all(extract_answer(b[0]) for b in beam):
                break

        # Final Selection
        final_selection = None
        status = "failed"

        if all_finished_paths:
            # 重複排除
            unique = {}
            for p in all_finished_paths:
                unique[p[0]] = p
            candidates = list(unique.values())
            candidates.sort(key=lambda x: x[1], reverse=True)
            final_selection = candidates[0]
            status = "success"
        elif beam:
            # タイムアウト時
            beam.sort(key=lambda x: x[1], reverse=True)
            final_selection = beam[0][:4] 
            status = "timeout_no_answer"
        else:
            return None, -999, [], "failed"

        return final_selection[0], final_selection[1], final_selection[2], status
# ==========================================
# 3. Main Logic
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--prm_model", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="data/result_strict.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_gen_loops", type=int, default=25)
    parser.add_argument("--num_candidates", type=int, default=3)
    # GPUメモリ管理: GeneratorとPRMを同居させるため調整が必要
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--max_model_len", type=int, default=4096)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"=== Strict Adaptive Beam Search ===")
    print(f"Merge Threshold: {MIN_STEP_CHARS} chars")
    print(f"Generator: {args.gen_model}")
    print(f"PRM: {args.prm_model}")

    # 1. Load PRM (HF)
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_model)
    if prm_tokenizer.pad_token is None: prm_tokenizer.pad_token = prm_tokenizer.eos_token
    prm_model = AutoModelForSequenceClassification.from_pretrained(
        args.prm_model, num_labels=1, torch_dtype=torch.bfloat16, device_map=device
    )
    prm_model.eval()

    # 2. Load Generator (vLLM)
    llm = LLM(
        model=args.gen_model, 
        tensor_parallel_size=1, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        dtype="bfloat16", 
        enforce_eager=True,
        max_model_len=args.max_model_len
    )
    # プロンプト作成用トークナイザ
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model)

    searcher = StrictAdaptiveBeamSearcher(llm, prm_model, prm_tokenizer, device, args.seed)

    # 3. Load Data
    dataset = load_dataset("HuggingFaceH4/math-500", split="test")

    results = []
    correct_count = 0
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Searching"):
        problem = item["problem"]
        gold_answer = extract_answer(item["solution"])
        
        # Chat Template適用
        messages = [
            {"role": "system", "content": "Please reason step by step and put your final answer within \\boxed{}."},
            {"role": "user", "content": problem}
        ]
        formatted_problem = gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # サーチ実行
        best_text, best_score, history, status = searcher.search(
            formatted_problem,
            beam_width=args.beam_width, 
            max_gen_loops=args.max_gen_loops, 
            num_candidates=args.num_candidates
        )
        
        pred_answer = extract_answer(best_text)
        is_correct = check_correctness(pred_answer, gold_answer)
        
        if is_correct: correct_count += 1
        
        results.append({
            "problem": problem,
            "gold": gold_answer,
            "pred": pred_answer,
            "is_correct": is_correct,
            "final_min_score": best_score,
            "status": status,
            "history": history,
            "full_text": best_text
        })

    acc = correct_count / len(dataset)
    print(f"\nFinal Accuracy: {acc:.2%}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed,
            "accuracy": acc,
            "config": vars(args),
            "details": results
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()