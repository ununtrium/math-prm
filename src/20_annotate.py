import argparse
import json
import os
import math
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp

# 相対インポート: src/utils.py を読み込む
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import (
    split_text_into_steps, 
    is_correct, 
    extract_answer_content
)

# ==========================================
# 1. 設定・定数
# ==========================================
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ★修正: 二重改行を防ぐため、先頭の \n を削除
# 文脈結合時に "\n" を明示的に管理する方が安全です
#TRIGGER_PHRASE = "The final answer is \\boxed{"
TRIGGER_PHRASE = ""  

# ==========================================
# 2. 確率計算ロジック
# ==========================================
def calculate_joint_log_prob(model, tokenizer, context_texts, target_answer_with_brace, device):
    """
    確率(0~1)ではなく、対数確率(-inf ~ 0)を返す。
    """
    # ★修正: 結合時に改行を明示的に入れる
    # context が空でない限り、改行を入れて接続する
    full_texts = []
    prefix_texts = []
    
    for ctx in context_texts:
        # 文脈の末尾に改行がなければ追加して TRIGGER を繋ぐ
        #separator = "\n" if ctx and not ctx.endswith("\n") else ""
        separator = ""
        
        full_text = ctx + separator + TRIGGER_PHRASE + target_answer_with_brace
        prefix_text = ctx + separator + TRIGGER_PHRASE
        
        full_texts.append(full_text)
        prefix_texts.append(prefix_text)
    
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, padding_side="right").to(device)
    prefix_inputs = tokenizer(prefix_texts, return_tensors="pt", padding=True, padding_side="right")
    prefix_lengths = prefix_inputs.attention_mask.sum(dim=1).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    
    log_probs = []
    for i in range(len(full_texts)):
        start_idx = prefix_lengths[i] - 1
        end_idx = inputs.attention_mask[i].sum() - 1
        
        # 異常系: Prefixだけで入力長を超えてしまった場合など
        if start_idx >= end_idx:
            log_probs.append(-float('inf')) 
            continue
            
        ans_logits = shift_logits[i, start_idx:end_idx, :]
        ans_labels = shift_labels[i, start_idx:end_idx]
        
        token_log_probs = torch.log_softmax(ans_logits, dim=-1)
        target_token_log_probs = token_log_probs.gather(1, ans_labels.unsqueeze(1)).squeeze(1)
        
        total_log_prob = target_token_log_probs.sum().item()
        log_probs.append(total_log_prob)
        
    return log_probs

# ==========================================
# 3. ワーカープロセス
# ==========================================
def worker_process(rank, gpu_id, lines_chunk, output_temp_file, args):
    print(f"[GPU {gpu_id}] Worker started for model: {args.model_id}")
    device = f"cuda:{gpu_id}"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=DTYPE, 
        device_map=device, 
        attn_implementation="sdpa"
    )
    model.eval()

    annotated_data = []
    iterator = tqdm(lines_chunk, desc=f"GPU {gpu_id}", position=rank) if len(lines_chunk) > 0 else []

    for line in iterator:
        try: 
            record = json.loads(line)
            source_id = record["source_id"]
            problem = record["problem"]
            ground_truth_raw = record["ground_truth"]
            paths = record["generated_paths"]
        except: 
            continue
            
        default_target_content = extract_answer_content(ground_truth_raw)
        if not default_target_content: continue

        for path_idx, path in enumerate(paths):
            path_id = f"{source_id}_path_{path_idx}"
            
            steps = split_text_into_steps(path)
            if not steps: continue

            # --- 正解判定 & Target決定 ---
            is_correct_outcome = False
            target_content = default_target_content
            
            generated_answer_content = extract_answer_content(path)
            
            if generated_answer_content:
                if is_correct(path, ground_truth_raw):
                    is_correct_outcome = True
                    target_content = generated_answer_content
                else:
                    is_correct_outcome = False

            # ★ターゲット構築: 数値 + "}"
            #probing_target = target_content + "}"
            probing_target = target_content

            # --- コンテキスト構築 ---
            context_list_str = []
            history_steps_list = []

            # Step 0: 問題文のみ
            current_text = problem
            context_list_str.append(current_text)
            history_steps_list.append([problem])

            # Step 1 ~ N
            for step in steps:
                # ★修正: ループ内では素直に改行で繋ぐ
                current_text += "\n" + step 
                context_list_str.append(current_text)
                new_history = history_steps_list[-1] + [step]
                history_steps_list.append(new_history)
            
            # 確率計算 (Batch処理)
            step_log_probs = []
            for i in range(0, len(context_list_str), args.batch_size):
                batch_contexts = context_list_str[i : i + args.batch_size]
                
                batch_log_probs = calculate_joint_log_prob(
                    model, tokenizer, batch_contexts, probing_target, device
                )
                step_log_probs.extend(batch_log_probs)
            
            # データ保存 (t=1 から保存)
            for t in range(1, len(step_log_probs)):
                current_log_prob = step_log_probs[t]
                prev_log_prob = step_log_probs[t-1]
                
                # ★重要: Inf対策
                # エラー等で -inf が返ってきた場合の安全策
                if current_log_prob == -float('inf') or prev_log_prob == -float('inf'):
                    log_prob_delta = 0.0
                else:
                    log_prob_delta = current_log_prob - prev_log_prob
                
                try:
                    raw_prob = math.exp(current_log_prob)
                    prev_raw_prob_val = math.exp(prev_log_prob)
                except OverflowError:
                    raw_prob = 0.0
                    prev_raw_prob_val = 0.0
                raw_prob_delta = raw_prob - prev_raw_prob_val
                step_idx = t 
                
                new_rec = {
                    "model_id": args.model_id,
                    "source_id": source_id,
                    "path_id": path_id,
                    "step_index": step_idx,
                    "step_text": steps[t-1],
                    "full_text_list": history_steps_list[t],
                    "log_prob": current_log_prob,
                    "log_prob_delta": log_prob_delta,
                    "raw_prob": raw_prob,
                    "raw_prob_delta": raw_prob_delta,
                    "is_outcome_correct": is_correct_outcome
                }
                annotated_data.append(new_rec)

    with open(output_temp_file, "w", encoding="utf-8") as f:
        for item in annotated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[GPU {gpu_id}] Finished.")

# メイン関数は変更なしでOK
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Evaluator Model ID")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL from generate.py")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSONL")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for probing")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} not found.")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: num_gpus = 1
    
    print(f"Starting Annotation with {args.model_id} on {num_gpus} GPUs...")

    with open(args.input_file, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    chunk_size = math.ceil(len(all_lines) / num_gpus)
    chunks = [all_lines[i:i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
    
    processes = []
    temp_files = []
    mp.set_start_method('spawn', force=True)

    for rank, chunk in enumerate(chunks):
        if not chunk: continue
        gpu_id = rank % num_gpus
        temp_file = f"{args.output_file}.part{rank}"
        temp_files.append(temp_file)
        
        p = mp.Process(target=worker_process, args=(rank, gpu_id, chunk, temp_file, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print("Merging output files...")
    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(temp_file)
    
    print(f"Done! Saved to {args.output_file}")

if __name__ == "__main__":
    main()