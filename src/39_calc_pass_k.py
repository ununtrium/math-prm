import argparse
import json
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Pass@1 and Pass@N")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the JSON file containing generated samples.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        return

    print(f"Loading {args.input_file} ...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # データ構造の吸収
    items = data.get("details", []) if isinstance(data, dict) else data

    if not items:
        print("No data found.")
        return

    total_problems = len(items)
    solved_at_least_once = 0
    total_samples = 0
    total_correct_samples = 0

    # パスごとの正解数分布を記録
    correct_counts_distribution = []

    for item in items:
        samples = item.get("generated_samples", [])
        if not samples:
            continue
        
        # 各サンプルの正誤を取得
        # (前回のデータ形式に基づき 'is_correct' キーを参照)
        is_correct_list = [s.get("is_correct", False) for s in samples]
        
        # この問題における正解数
        correct_count = sum(is_correct_list)
        n_samples = len(samples)

        # 統計更新
        if correct_count > 0:
            solved_at_least_once += 1
        
        total_correct_samples += correct_count
        total_samples += n_samples
        correct_counts_distribution.append(correct_count)

    # 計算
    # Pass@1: 全生成パスに対する正解パスの割合（期待値）
    pass_at_1 = total_correct_samples / total_samples if total_samples > 0 else 0.0
    
    # Pass@N (Empirical): 少なくとも1つ正解が含まれていた問題の割合
    # ここでは N=16 (データに含まれる全サンプル数) とします
    pass_at_n = solved_at_least_once / total_problems if total_problems > 0 else 0.0

    print("-" * 40)
    print(f"Total Problems: {total_problems}")
    print(f"Total Samples Generated: {total_samples}")
    print("-" * 40)
    print(f"Pass@1 (Avg Accuracy):  {pass_at_1:.2%}")
    print(f"Pass@16 (Solvability):  {pass_at_n:.2%}")
    print("-" * 40)
    
    # 追加分析: 正解数の分布
    print("Correct Path Count Distribution (per problem):")
    counts = np.array(correct_counts_distribution)
    print(f"  0 correct (Impossible for BoN): {np.sum(counts == 0)} problems ({(np.sum(counts == 0)/total_problems):.1%})")
    print(f"  1-{n_samples-1} correct (BoN/MV needed): {np.sum((counts > 0) & (counts < 16))} problems")
    print(f"  All {n_samples} correct (Easy):       {np.sum(counts == 16)} problems")

if __name__ == "__main__":
    main()