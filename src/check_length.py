import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DATA_FILE = "data/annotated_train_data.jsonl"
CHECK_LIMIT = 2048  # 学習時の設定と同じにする

def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"Scanning {DATA_FILE}...")
    lengths = []
    over_limit_count = 0
    total_count = 0
    
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Tokenizing"):
        try:
            record = json.loads(line)
            text = record["full_text"]
            
            # トークン化して長さを測る (truncationなし)
            tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
            length = len(tokens)
            
            lengths.append(length)
            total_count += 1
            
            if length > CHECK_LIMIT:
                over_limit_count += 1
        except:
            continue

    lengths = np.array(lengths)
    
    # ==========================================
    # 結果表示
    # ==========================================
    print("\n" + "="*30)
    print("TOKEN LENGTH REPORT")
    print("="*30)
    print(f"Total Samples: {total_count}")
    print(f"Over {CHECK_LIMIT} tokens: {over_limit_count} ({over_limit_count/total_count:.1%})")
    print("-" * 20)
    print(f"Min:    {lengths.min()}")
    print(f"Mean:   {lengths.mean():.1f}")
    print(f"Median: {np.median(lengths):.1f}")
    print(f"Max:    {lengths.max()}")
    print(f"95%ile: {np.percentile(lengths, 95):.1f}")
    print(f"99%ile: {np.percentile(lengths, 99):.1f}")
    
    # ヒストグラム保存
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(CHECK_LIMIT, color='red', linestyle='dashed', linewidth=2, label=f'Limit ({CHECK_LIMIT})')
    plt.title('Token Length Distribution')
    plt.xlabel('Token Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("data/token_length_dist.png")
    print("\nHistogram saved to data/token_length_dist.png")

    # ==========================================
    # 診断とアドバイス
    # ==========================================
    print("\n[DIAGNOSIS]")
    if over_limit_count / total_count > 0.05: # 5%以上超えている場合
        print(f"🔴 DANGER: {over_limit_count} samples are truncated!")
        print("-> 'truncation_side=left' fix is HIGHLY RECOMMENDED.")
        print("-> Or increase MAX_SEQ_LENGTH if GPU memory allows.")
    elif over_limit_count > 0:
        print("🟡 WARNING: Some samples are truncated.")
        print("-> Checking predictions for these specific long samples might reveal the issue.")
    else:
        print("🟢 SAFE: No samples exceed the limit.")
        print("-> Truncation is NOT the cause of low correlation.")
        print("-> Look into: Learning Rate (too high?), Epochs (too few?), or Model Capacity.")

if __name__ == "__main__":
    main()