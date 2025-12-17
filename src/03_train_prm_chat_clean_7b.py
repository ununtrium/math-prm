import os
import torch
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)
from datasets import load_dataset

# ==========================================
# 1. 設定パラメータ (Configuration)
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-Math-7B-Instruct" 
TRAIN_FILE = "data/p_scaled_value_train_30k_chat.jsonl"
OUTPUT_DIR = "models/prm_7b_30k_v3.0_chat_clean_new"

# Max Sequence Length
MAX_SEQ_LENGTH = 3072     

# ★大規模データ(2M samples)向けチューニング
LEARNING_RATE = 2e-5      # ★変更: データが多いので少し下げて丁寧に学習させる(1e-5 or 2e-5)
NUM_EPOCHS = 1            # ★変更: 2Mサンプルなら1エポックで十分収束します
BATCH_SIZE = 2            
GRAD_ACCUMULATION = 16    # ★変更: 勾配を安定させるため、実質バッチサイズを大きめに確保 (4*16*N_GPU)
WARMUP_RATIO = 0.0       # ★変更: 標準的な値に戻す
SEED = 42

# ==========================================
# タグ削除関数 (変更なし・必須)
# ==========================================
def clean_text_for_prm(text):
    text = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>\n?", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>user\n?", "", text)
    text = re.sub(r"<\|im_end\|>\n?<\|im_start\|>assistant\n?", "\n", text)
    text = text.replace("<|im_end|>", "")
    return text.strip()

def main():
    set_seed(SEED)
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs. Training on 2M+ samples (1 Epoch)...")

    # ==========================================
    # 2. データセット (変更なし)
    # ==========================================
    # JSONLが巨大な場合、メモリ不足なら streaming=True を検討してください
    # (64GB以上のRAMがあれば通常のload_datasetで大丈夫です)
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    
    # 検証データは絶対数として2000件あれば統計的に十分信頼できます
    dataset = dataset.train_test_split(test_size=5000, seed=SEED)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    
    print(f"Train samples: {len(train_ds)} (approx 2M)")
    print(f"Eval samples:  {len(eval_ds)}")

    # ==========================================
    # 3. 前処理 (変更なし)
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        cleaned_texts = [clean_text_for_prm(text) for text in examples["full_text"]]
        tokenized = tokenizer(
            cleaned_texts,
            padding=False,          
            truncation=True,        
            max_length=MAX_SEQ_LENGTH
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    remove_cols = train_ds.column_names
    
    # 200万件の処理には時間がかかるので num_proc をCPUコア数に合わせて増やしてください
    print("Tokenizing... (This may take a while for 2M samples)")
    train_ds = train_ds.map(preprocess_function, batched=True, num_proc=16, remove_columns=remove_cols)
    eval_ds = eval_ds.map(preprocess_function, batched=True, num_proc=4, remove_columns=remove_cols)

    # ==========================================
    # 4. モデル (変更なし)
    # ==========================================
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        torch_dtype=torch.bfloat16, 
        device_map=None,            
        attn_implementation="flash_attention_2" 
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "regression"

    # ==========================================
    # 5. 学習設定 (Trainer)
    # ==========================================
    os.environ["WANDB_PROJECT"] = "Delta-PRM"
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        
        # ★2Mデータ用の保存設定
        eval_strategy="steps",
        eval_steps=1000,             # 2000ステップごとに評価
        save_strategy="steps",
        save_steps=1000,             # 2000ステップごとに保存
        save_total_limit=3,

        load_best_model_at_end=True, 
        metric_for_best_model="loss",
        
        logging_steps=50,
        bf16=True,                  
        dataloader_num_workers=4,
        group_by_length=True,
        run_name="prm_7b_run_30k_v3.0_chat_clean_new",
        report_to="wandb",          
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Done!")

if __name__ == "__main__":
    main()