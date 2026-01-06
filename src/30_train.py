import argparse
import os
import sys
import math
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

# ==========================================
# 1. 引数設定
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train PRM with Log-Sigmoid Reward")
    
    # パス設定
    parser.add_argument("--train_file", type=str, required=True, help="Path to annotated jsonl file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/prm_final")
    
    # ★重要変更: 分布に基づいた報酬パラメータ
    # data distribution: Correct Median ~ 0.037 (log -3.3), Incorrect Median ~ 0.0001 (log -9.2)
    # Threshold (tau) should be around -5.0 to separate them.
    
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="Scale for sigmoid. 1.0 is standard.")
    parser.add_argument("--tau", type=float, default=-5.0, 
                        help="Center of sigmoid. Based on data distribution (log space).")
    
    # Delta (aux_weight) は分布が悪かったため 0.0 (不使用) を推奨
    parser.add_argument("--aux_weight", type=float, default=0.0, help="Weight for Delta term.")
    
    parser.add_argument("--beta", type=float, default=0.0, help="Penalty for incorrect paths.")
    
    # 学習ハイパーパラメータ
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--test_size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="Delta-PRM")
    parser.add_argument("--run_name", type=str, default="prm_run")
    
    return parser.parse_args()

# ==========================================
# 2. 報酬計算ロジック (Log-Sigmoid)
# ==========================================
def stable_sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def calculate_reward(log_prob, log_prob_delta, is_correct, alpha, tau, aux_weight, beta):
    """
    Reward = Sigmoid( alpha * (LogProb - tau) ) + aux_weight * Tanh(Delta)
    
    分布が 0.0 に張り付いているため、Raw Probではなく Log Prob を Sigmoid で広げて使う。
    """
    # 1. Main: Log Probability based Score
    # LogProb (-100 ~ 0) を Sigmoid で 0~1 にマッピングする
    # tau=-5.0 の場合:
    #   LogProb=-3.3 (Correct Median) -> Sigmoid(1.7)  ≈ 0.84 (高スコア)
    #   LogProb=-9.2 (Incorrect Median) -> Sigmoid(-4.2) ≈ 0.01 (低スコア)
    # これで明確な差がつく。
    
    safe_log_prob = max(log_prob, -100.0)
    logit = alpha * (safe_log_prob - tau)
    main_score = stable_sigmoid(logit)

    # 2. Aux: Delta (今回は使わない想定だがロジックは残す)
    if aux_weight != 0.0:
        delta_val = log_prob_delta if log_prob_delta is not None else 0.0
        delta_val = max(min(delta_val, 50.0), -50.0)
        aux_bonus = aux_weight * math.tanh(delta_val)
    else:
        aux_bonus = 0.0
    
    base_reward = main_score + aux_bonus
    
    # 3. Penalty
    if is_correct:
        final_label = base_reward
    else:
        final_label = base_reward - beta
        
    return final_label

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print(f"Loading data from {args.train_file}...")
    print(f"--- Reward Params (Log-Sigmoid) ---")
    print(f"  Formula: Sigmoid( {args.alpha} * (LogProb - ({args.tau})) )")
    print(f"-----------------------------------")

    # 1. データセット
    full_dataset = load_dataset("json", data_files=args.train_file, split="train")
    
    if len(full_dataset) > args.test_size:
        dataset_split = full_dataset.train_test_split(test_size=args.test_size, seed=args.seed)
    else:
        dataset_split = full_dataset.train_test_split(test_size=0.05, seed=args.seed)
        
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples:  {len(eval_dataset)}")

    # 2. トークナイザ
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 前処理
    def preprocess_function(examples):
        inputs = []
        labels = []
        
        history_lists = examples["full_text_list"]
        log_probs = examples["log_prob"]
        # log_prob_delta を取得 (無い場合は0)
        deltas = examples.get("log_prob_delta", [0.0] * len(log_probs))
        if deltas is None: deltas = [0.0] * len(log_probs)

        is_corrects = examples["is_outcome_correct"]
        
        for i, history in enumerate(history_lists):
            text = "\n".join(history)
            inputs.append(text)
            
            label_val = calculate_reward(
                log_prob=log_probs[i],
                log_prob_delta=deltas[i],
                is_correct=is_corrects[i],
                alpha=args.alpha,
                tau=args.tau, # 分布から決定した閾値
                aux_weight=args.aux_weight,
                beta=args.beta
            )
            labels.append(float(label_val))
            
        model_inputs = tokenizer(
            inputs, 
            max_length=args.max_length, 
            truncation=True, 
            padding=False 
        )
        model_inputs["labels"] = labels
        return model_inputs

    print("Tokenizing...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=full_dataset.column_names, num_proc=16)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=full_dataset.column_names, num_proc=4)

    # 4. モデル
    print(f"Loading model: {args.base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "eager"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # 5. Metrics
    # 5. Metrics (修正版)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        
        # MSE (Regressionの基本)
        mse = ((predictions - labels) ** 2).mean().item()
        
        # Pearson Correlation
        try:
            pearson = np.corrcoef(predictions, labels)[0, 1]
        except:
            pearson = 0.0
            
        # ★追加: AUC Calculation
        # PRMのラベルは連続値(0.01, 0.85等)なので、AUC計算用に
        # 「0.5より大きければ正解(1)、そうでなければ不正解(0)」とみなして2値化します。
        # ORMの場合は元々0か1なので、この処理をしてもしなくても結果は変わりません。
        binary_targets = (labels > 0.5).astype(int)
        
        # 評価バッチ内に「すべて0」や「すべて1」しかないとAUC計算でエラーになるためTry-Except
        try:
            # uniqueなクラスが2つ以上ある場合のみ計算
            if len(np.unique(binary_targets)) > 1:
                auc = roc_auc_score(binary_targets, predictions)
            else:
                auc = 0.5 # 計算不能時はランダム相当とする
        except:
            auc = 0.5

        return {"mse": mse, "pearson": pearson, "auc": auc}

    # 6. Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        
        logging_steps=50,
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="wandb",
        run_name=args.run_name,
        dataloader_num_workers=4,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()