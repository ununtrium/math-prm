import argparse
import json
import os
import torch
import glob
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# デフォルト設定
# ==========================================
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 3072
# アノテーションコードの設定に合わせる
DEFAULT_STEP_MERGE_CHARS = 50
DEFAULT_TARGET_MAX_STEPS = 15

# ==========================================
# ユーティリティ: ステップ分割 (アノテーションコードと統一)
# ==========================================
def reduce_step_count(steps, target_max=15, min_chars=50):
    if len(steps) <= target_max: return steps
    merged_steps = []
    buffer = ""
    for i, step in enumerate(steps):
        if i == 0: buffer = step; continue
        if len(step) < min_chars or len(buffer) < min_chars: buffer += "\n" + step
        else: merged_steps.append(buffer); buffer = step
    if buffer: merged_steps.append(buffer)
    while len(merged_steps) > target_max:
        new_merged = []
        for i in range(0, len(merged_steps), 2):
            if i + 1 < len(merged_steps): new_merged.append(merged_steps[i] + "\n" + merged_steps[i+1])
            else: new_merged.append(merged_steps[i])
        merged_steps = new_merged
        if len(merged_steps) <= 1: break
    return merged_steps

def get_adaptive_steps(path_text, target_max=15, min_merge_chars=50):
    # 1. まずダブル改行(段落)で分割を試みる
    steps = [s.strip() for s in re.split(r'\n\s*\n', path_text) if s.strip()]
    
    # 2. 分割数が少なすぎる(3未満)場合は、単一改行で分割し直す
    if len(steps) < 3:
        alt_steps = [s.strip() for s in path_text.split('\n') if s.strip()]
        if len(alt_steps) >= 3:
            steps = alt_steps
            
    # 3. 分割が多すぎる場合はマージ処理
    if len(steps) > target_max:
        # アノテーションコードに合わせて raw_lines から再構築
        raw_lines = [s.strip() for s in path_text.split('\n') if s.strip()]
        steps = reduce_step_count(raw_lines, target_max=target_max, min_chars=min_merge_chars)
        
    # 空リスト対策 (念のため)
    if not steps and path_text.strip():
        steps = [path_text.strip()]
        
    return steps

def parse_args():
    parser = argparse.ArgumentParser(description="Step-wise PRM Rescoring")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file or directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save rescored JSON files.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained PRM model.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser.parse_args()

# ==========================================
# メイン処理
# ==========================================
def main():
    args = parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input path {args.input_path} not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input_path):
        files = [args.input_path]
    else:
        files = glob.glob(os.path.join(args.input_path, "*.json"))
    
    if not files:
        print(f"No JSON files found in {args.input_path}")
        return
    
    print(f"Found {len(files)} files.")
    print(f"Loading Model: {args.model_path} ...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, 
        num_labels=1, 
        torch_dtype=dtype, 
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    for fpath in files:
        fname = os.path.basename(fpath)
        print(f"\nProcessing {fname}...")
        
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "details" in data:
            items = data["details"]
        else:
            items = data 
            
        rescored_items = []
        
        for item in tqdm(items, desc="Rescoring"):
            problem = item["problem"]
            
            if "generated_samples" in item:
                raw_paths = [s["text"] for s in item["generated_samples"]]
            elif "paths" in item:
                raw_paths = item["paths"]
            else:
                raw_paths = []

            new_path_scores = []
            new_step_scores = []
            
            for path in raw_paths:
                # ★修正: アノテーションコードと同じ分割関数を使用
                steps = get_adaptive_steps(
                    path, 
                    target_max=DEFAULT_TARGET_MAX_STEPS, 
                    min_merge_chars=DEFAULT_STEP_MERGE_CHARS
                )
                
                if not steps:
                    new_path_scores.append(-99.0)
                    new_step_scores.append([-99.0])
                    continue
                
                # 入力作成 (Raw形式: Problem + \n + Step)
                # アノテーションコードの context_list 作成ロジックと合わせる
                # annotation: current_text += step + "\n" -> context_list.append(current_text)
                # evaluation: problem + \n + step1 + \n + step2 ...
                
                step_inputs = []
                curr_text = problem.strip()
                
                # アノテーションコードではChat templateを使っているが、
                # PRM学習時の入力形式(Raw text without template tags)に合わせる必要がある。
                # 前回の議論通り「タグ削除済み(Raw)」で学習したのであれば、単純結合でOK。
                for step in steps:
                    curr_text += "\n" + step
                    step_inputs.append(curr_text)
                
                current_rewards = []
                with torch.no_grad():
                    for i in range(0, len(step_inputs), args.batch_size):
                        batch = step_inputs[i : i+args.batch_size]
                        inputs = tokenizer(
                            batch, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True, 
                            max_length=args.max_length
                        ).to(model.device)
                        
                        out = model(**inputs)
                        current_rewards.extend(out.logits.squeeze(-1).tolist())
                
                # 集計 (Min)
                final_score = min(current_rewards) if current_rewards else -99.0
                
                new_path_scores.append(final_score)
                new_step_scores.append(current_rewards)
            
            new_item = item.copy()
            new_item["scores_new"] = new_path_scores
            new_item["step_scores_new"] = new_step_scores
            
            if "generated_samples" in new_item:
                for idx, sample in enumerate(new_item["generated_samples"]):
                    if idx < len(new_step_scores):
                        sample["step_scores"] = new_step_scores[idx]
                        sample["final_score"] = new_path_scores[idx]

            rescored_items.append(new_item)
            
        if "details" in data:
            data["details"] = rescored_items
            output_data = data
        else:
            output_data = rescored_items

        if isinstance(output_data, dict) and "config" in output_data:
            output_data["config"]["prm_model_eval"] = args.model_path

        out_path = os.path.join(args.output_dir, fname)
        print(f"Saving to {out_path}...")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\nDone! All files processed.")

if __name__ == "__main__":
    main()