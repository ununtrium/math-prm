import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# 学習済みモデルのパス
MODEL_PATH = "models/delta_prm_1.5b_pre_v1" # または checkpoint-400 など
DATA_PATH = "data/annotated_train_data.jsonl"

def main():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        num_labels=1,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("Loading validation data...")
    # 検証用に一部だけロード
    dataset = load_dataset("json", data_files=DATA_PATH, split="train[:100]")
    
    preds = []
    labels = []
    
    print("Running inference...")
    for item in tqdm(dataset):
        text = item["full_text"]
        label = item["label"]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            output = model(**inputs)
            score = output.logits.item()
            
        preds.append(score)
        labels.append(label)

    # 評価
    preds = np.array(preds)
    labels = np.array(labels)
    
    mse = np.mean((preds - labels)**2)
    correlation = np.corrcoef(preds, labels)[0, 1]
    
    print("\n" + "="*30)
    print("DIAGNOSTIC REPORT")
    print("="*30)
    print(f"MSE Loss (Recalculated): {mse:.4f}")
    print(f"Correlation (Pearson):   {correlation:.4f}")
    print("-" * 20)
    print(f"Label Range:      Min={labels.min():.2f}, Max={labels.max():.2f}, Mean={labels.mean():.2f}")
    print(f"Prediction Range: Min={preds.min():.2f}, Max={preds.max():.2f}, Mean={preds.mean():.2f}")
    print("-" * 20)
    print("Sample Predictions:")
    for i in range(5):
        print(f"  Label: {labels[i]:.2f}  |  Pred: {preds[i]:.2f}  |  Diff: {abs(labels[i]-preds[i]):.2f}")

if __name__ == "__main__":
    main()