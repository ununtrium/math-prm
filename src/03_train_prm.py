import os
import torch
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
# モデル設定
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct" 

# データパス
TRAIN_FILE = "data/p_scaled_value_train_30k.jsonl"
OUTPUT_DIR = "models/prm_1.5b_30k_v3.0"

# ★変更点: 長文対応設定★
# 分析結果(Max 2754 tokens)に基づき、3072に拡張して切り捨てを回避
MAX_SEQ_LENGTH = 3072     

# 学習ハイパーパラメータ
LEARNING_RATE = 2e-5      
NUM_EPOCHS = 2            # 学習不足を防ぐため3周回す
BATCH_SIZE = 4            # GPUメモリがきつい場合は 2 に下げてください
GRAD_ACCUMULATION = 8     # 実質バッチサイズ = 4 * 8GPU * 4 = 128
WARMUP_RATIO = 0.03
SEED = 42

def main():
    set_seed(SEED)
    
    # GPU環境の確認
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs. Preparing for training...")
    print(f"Max Sequence Length: {MAX_SEQ_LENGTH}")

    # ==========================================
    # 2. データセットの準備
    # ==========================================
    print(f"Loading dataset from {TRAIN_FILE}...")
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    
    # 学習用(95%)と検証用(5%)に分割
    dataset = dataset.train_test_split(test_size=5000, seed=SEED)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Eval samples:  {len(eval_ds)}")

    # ==========================================
    # 3. トークナイザと前処理
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ※補足: MAX_SEQ_LENGTH = 3072 にしたため、
    # truncation_side='left' のハックは不要になりました（デフォルトのままでOK）

    def preprocess_function(examples):
        # 入力: full_text (問題文 + ... + ステップ)
        # ラベル: label (スコア)
        tokenized = tokenizer(
            examples["full_text"],
            padding=False,          # DataCollatorで動的にパディング
            truncation=True,        # 3072超えがあれば切り捨てる(発生しないはず)
            max_length=MAX_SEQ_LENGTH
        )
        tokenized["labels"] = examples["label"]
        return tokenized

    # 不要なカラム（分析用メタデータ）を削除して軽量化
    remove_cols = train_ds.column_names
    
    print("Tokenizing training dataset...")
    train_ds = train_ds.map(
        preprocess_function, 
        batched=True, 
        num_proc=8, 
        remove_columns=remove_cols,
        desc="Tokenizing train"
    )
    
    print("Tokenizing evaluation dataset...")
    eval_ds = eval_ds.map(
        preprocess_function, 
        batched=True, 
        num_proc=4, 
        remove_columns=remove_cols,
        desc="Tokenizing eval"
    )

    # ==========================================
    # 4. モデルの準備 (Regression Head)
    # ==========================================
    print("Loading model...")
    # num_labels=1 -> 回帰モード(MSELoss)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        torch_dtype=torch.bfloat16, 
        device_map=None,            # DDP用
        attn_implementation="flash_attention_2" 
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "regression"

    # ==========================================
    # 5. 学習設定 (Trainer)
    # ==========================================
    # WandB設定
    os.environ["WANDB_PROJECT"] = "Delta-PRM"
    os.environ["WANDB_LOG_MODEL"] = "false"
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        
        # 評価・保存設定
        eval_strategy="steps",
        eval_steps=1000,             # こまめにLossを確認
        save_strategy="steps",
        save_steps=1000,             
        save_total_limit=5,         # 保存数を少し増やす
        logging_steps=100,
        
        # 精度・高速化設定
        bf16=True,                  
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        group_by_length=True,       # 長さが近いデータをまとめてパディング削減
        
        # ログ設定
        run_name="prm_1.5b_run_30k_v3.0",
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

    # ==========================================
    # 6. 学習実行
    # ==========================================
    print("Starting training...")
    trainer.train()

    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    metrics = trainer.evaluate()
    print("Final Eval Metrics:", metrics)

if __name__ == "__main__":
    main()