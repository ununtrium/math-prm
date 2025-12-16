import os
import argparse
import json
import re
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vllm not found. Please install via: pip install vllm")
    exit()

try:
    from latex2sympy2 import latex2sympy
    from sympy import simplify
except ImportError:
    print("Warning: latex2sympy2 or sympy not found. Strict checking might be limited.")

# ==========================================
# ユーティリティ (正解判定)
# ==========================================
def extract_answer(text):
    if not text: return None
    # \boxed{...} を抽出。複数ある場合は最後を採用
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches:
        return matches[-1]
    
    # \boxedがない場合のフォールバック（The answer is ... 形式など）
    # 必要に応じて調整してください。今回はMATHデータセット前提でboxed重視。
    return None

def robust_float_check(pred, gold):
    try:
        def clean_to_float(s):
            s = str(s).replace(r"\frac", "").replace("{", "(").replace("}", ")").replace("^", "**")
            s = s.replace(r"\left", "").replace(r"\right", "").replace(",", "").replace("$", "")
            return float(eval(s))
        
        # 文字列が含まれていない場合のみ数値比較
        if not any(c.isalpha() for c in str(pred)) and not any(c.isalpha() for c in str(gold)):
            return abs(clean_to_float(pred) - clean_to_float(gold)) < 1e-6
    except:
        pass
    return False

def check_correctness(pred_str, gold_str):
    if not pred_str or not gold_str: return False
    pred_str = str(pred_str).strip()
    gold_str = str(gold_str).strip()
    
    # 1. 完全一致
    if pred_str == gold_str: return True
    
    # 2. SymPyによる数式等価性判定
    try:
        sym_pred = latex2sympy(pred_str)
        sym_gold = latex2sympy(gold_str)
        if simplify(sym_pred - sym_gold) == 0: return True
    except:
        pass
    
    # 3. 数値誤差許容判定
    return robust_float_check(pred_str, gold_str)

# ==========================================
# メイン処理
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output json")
    
    # 実験パラメータ
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--n", type=int, default=20, help="Number of samples per problem (Majority Voting N)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    
    # システム設定
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # 1. データセット読み込み
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceH4/math-500", split="test")
    
    # 2. vLLMエンジンの初期化
    print(f"Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        enforce_eager=True,
        seed=args.seed
    )
    
    # SamplingParamsの設定
    # n=args.n にすることで、1つのプロンプトに対してN個の出力を生成します
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed
    )

    # 3. プロンプト作成 (Qwen-Math形式)
    # チャットテンプレートを適用するのがベストですが、ここでは単純化のためInstruction形式で整形
    # 必要に応じて tokenizer.apply_chat_template を使用してください
    prompts = []
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    print("Preparing prompts...")
    for item in dataset:
        # Qwen-MathなどのChatモデル用のテンプレート適用
        messages = [
            {"role": "system", "content": "Please reason step by step and put your final answer within \\boxed{}."},
            {"role": "user", "content": item["problem"]}
        ]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(text)

    # 4. バッチ生成実行
    print(f"Generating {args.n} samples for {len(prompts)} problems...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. 集計と保存
    results = []
    correct_counts = 0
    
    print("Processing results...")
    for i, request_output in enumerate(tqdm(outputs)):
        problem_data = dataset[i]
        problem = problem_data["problem"]
        gold_solution = problem_data["solution"]
        gold_answer = extract_answer(gold_solution)
        
        generated_samples = []
        answers_pool = []
        
        # N個の出力それぞれについて処理
        for completion in request_output.outputs:
            generated_text = completion.text
            pred_answer = extract_answer(generated_text)
            
            # 正誤判定
            is_correct = check_correctness(pred_answer, gold_answer)
            
            # 保存用リストに追加 (Best-of-N用)
            generated_samples.append({
                "text": generated_text,
                "pred_answer": pred_answer,
                "is_correct": is_correct
            })
            
            if pred_answer is not None:
                answers_pool.append(pred_answer)
        
        # Majority Voting (多数決)
        if answers_pool:
            # 最頻値を抽出
            # 注意: "2" と "2.0" や "\frac{1}{2}" と "0.5" は文字列として別扱いになるため
            # 厳密には正規化が必要ですが、簡易的に文字列カウントを行います
            counter = Counter(answers_pool)
            majority_answer, count = counter.most_common(1)[0]
            majority_is_correct = check_correctness(majority_answer, gold_answer)
        else:
            majority_answer = None
            majority_is_correct = False
            
        if majority_is_correct:
            correct_counts += 1
            
        # 結果オブジェクト構築
        results.append({
            "problem": problem,
            "gold_solution": gold_solution,
            "gold_answer": gold_answer,
            "majority_answer": majority_answer,
            "majority_is_correct": majority_is_correct,
            "generated_samples": generated_samples  # ここに全生成データが入る
        })

    # 最終精度の計算
    accuracy = correct_counts / len(dataset)
    print(f"Majority Voting Accuracy (N={args.n}, seed={args.seed}): {accuracy:.2%}")

    # JSON保存
    output_data = {
        "config": {
            "model": args.model,
            "seed": args.seed,
            "n": args.n,
            "temperature": args.temperature
        },
        "accuracy": accuracy,
        "details": results
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()