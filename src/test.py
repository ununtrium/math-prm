import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# vLLMのインポート (生成用)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vllm not found. Please install it for faster generation.")

import utils  # utils.py をインポート

# ==========================================
# Configuration: Benchmark Paths
# ==========================================
# AIMEなどはHuggingFaceにある標準的なリポジトリを参照するように変更
BENCHMARK_CONFIG = {
    "math500": {
        "path": "HuggingFaceH4/math-500", 
        "split": "test", 
        "key_prob": "problem", 
        "key_ans": "answer"
    },
    "aime24": {
        "path": "HuggingFaceH4/aime_2024", # AIME 2024の標準的なセット
        "split": "train", # HFのデータセット構成による(trainしかない場合が多い)
        "key_prob": "problem", 
        "key_ans": "answer" # solutionやanswerなどデータセットによる
    },
    # AIME 2025など、HFにない場合はローカルファイルを指定可能
    "aime25": {
        "path": "math-ai/aime25", 
        "split": "test", # HFのデータセット構成による(trainしかない場合が多い)
        "key_prob": "problem", 
        "key_ans": "answer" # solutionやanswerなどデータセットによる
    },
}

def load_benchmark_data(bench_name):
    """ベンチマークデータをロードして統一形式のリストにする"""
    cfg = BENCHMARK_CONFIG.get(bench_name)
    
    # Configにない場合はパスとして扱う
    if not cfg:
        if os.path.exists(bench_name):
            print(f"Loading local file: {bench_name}")
            return load_dataset("json", data_files=bench_name, split="train")
        raise ValueError(f"Unknown benchmark: {bench_name}")
    
    print(f"Loading benchmark from: {cfg['path']}")
    
    # ローカルJSONLの場合
    if cfg.get("format") == "json":
        dataset = load_dataset("json", data_files=cfg["path"], split="train")
    else:
        # HuggingFace Hubの場合
        try:
            dataset = load_dataset(cfg["path"], split=cfg.get("split", "test"))
        except Exception as e:
            # split名が違う場合のフォールバック
            print(f"Warning: split '{cfg.get('split')}' failed. Trying 'train'...")
            dataset = load_dataset(cfg["path"], split="train")
    
    # キーを統一 ("problem", "gold_answer")
    standardized = []
    for item in dataset:
        # データセットによってカラム名が違うので吸収
        prob = item.get(cfg["key_prob"])
        ans = item.get(cfg["key_ans"])
        
        # answerがない場合(solutionなど)の対応
        if ans is None and "solution" in item:
            ans = item["solution"]
            
        standardized.append({
            "problem": prob,
            "gold_answer": ans
        })
    return standardized

# ==========================================
# 1. Generation Phase (Batch Processing & Incremental Save)
# ==========================================
def run_generation(args):
    if not VLLM_AVAILABLE:
        raise ImportError("vLLM is required. Please `pip install vllm`.")

    print(f"Loading Generator (vLLM): {args.generator_name_or_path}")
    
    llm = LLM(
        model=args.generator_name_or_path,
        tensor_parallel_size=1, 
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        max_model_len=4096 # メモリ節約のため明示的に制限しても良い
    )
    
    # Baseモデルかどうか判定
    is_instruct = "Instruct" in args.generator_name_or_path

    # ★修正1: Stop Tokenの強化 (Baseモデルの暴走防止)
    stop_tokens = ["<|im_end|>", "\n\nProblem:", "\n\nSolution:", "Problem:", "User:"]
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
        n=args.num_samples,
        stop=stop_tokens # 追加
    )

    tokenizer = AutoTokenizer.from_pretrained(args.generator_name_or_path)
    gen_model_name = os.path.basename(args.generator_name_or_path)

    # ★修正2: バッチサイズの縮小
    # samples=64なら、一度に処理する問題数を減らさないとメモリが死ぬ
    # samples=64 -> BATCH_SIZE=4 くらい推奨
    # samples=16 -> BATCH_SIZE=16 くらい推奨
    if args.num_samples >= 64:
        BATCH_SIZE = 4 
    elif args.num_samples >= 16:
        BATCH_SIZE = 16
    else:
        BATCH_SIZE = 32

    for bench_name in args.target_benchmarks:
        try:
            dataset = load_benchmark_data(bench_name)
        except Exception as e:
            print(f"Skipping {bench_name}: {e}")
            continue

        print(f"Benchmark: {bench_name} ({len(dataset)} problems)")

        for seed in args.seeds:
            print(f"  Running Seed: {seed} (Batch Size: {BATCH_SIZE})")
            sampling_params.seed = seed
            
            output_dir = os.path.join("generation_results", bench_name, gen_model_name, f"samples_{args.num_samples}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"seed_{seed}.jsonl")
            
            if os.path.exists(output_file) and not args.overwrite:
                print(f"    Skipping (exists): {output_file}")
                continue

            # ファイルを初期化（空にする）
            with open(output_file, "w", encoding="utf-8") as f:
                pass 

            # 全データを準備
            all_prompts = []
            all_data = [] # 元データを保持

            for item in dataset:
                problem = item["problem"]
                if is_instruct:
                    prompt_text = utils.build_prompt(tokenizer, problem)
                else:
                    # Baseモデル用
                    prompt_text = (
                        f"Problem:\n{problem}\n\n"
                        "Please reason step by step and put your final answer within \\boxed{}.\n\n"
                        "Solution:"
                    )
                all_prompts.append(prompt_text)
                all_data.append(item)

            # ★修正3: 明示的なバッチループと都度保存
            total_items = len(all_prompts)
            
            # tqdmで進捗を見えるようにする
            with tqdm(total=total_items, desc=f"    Generating {bench_name}") as pbar:
                for i in range(0, total_items, BATCH_SIZE):
                    batch_prompts = all_prompts[i : i + BATCH_SIZE]
                    batch_data = all_data[i : i + BATCH_SIZE]

                    # ここでvLLM生成実行（ここだけメモリを食う）
                    try:
                        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                    except Exception as e:
                        print(f"\nError in batch generation: {e}")
                        # エラーが出ても次のバッチへ進むようにする
                        continue

                    batch_results = []
                    for j, output in enumerate(outputs):
                        # 元データとの紐付け
                        original_item = batch_data[j]
                        problem = original_item["problem"]
                        gold = original_item["gold_answer"]
                        
                        responses_data = []
                        for sample in output.outputs:
                            text = sample.text
                            
                            extracted = utils.extract_answer_content(text)
                            gold_content = utils.extract_answer_content(gold) 
                            if gold_content is None: gold_content = gold
                            is_correct = utils.check_equivalence(extracted, gold_content)
                            steps = utils.split_text_into_steps(text)
                            
                            responses_data.append({
                                "text": text,
                                "steps": steps,
                                "extracted": extracted,
                                "is_correct": is_correct
                            })
                        
                        batch_results.append({
                            "problem": problem,
                            "gold_answer": gold,
                            "responses": responses_data
                        })

                    # ★バッチが終わるたびに追記保存 ('a' mode)
                    with open(output_file, "a", encoding="utf-8") as f:
                        for r in batch_results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    
                    # メモリ解放を明示的に行う
                    del outputs
                    del batch_results
                    import gc; gc.collect()
                    
                    pbar.update(len(batch_prompts))

            print(f"    Saved to {output_file}")
    
    # モデル解放
    import gc
    del llm
    gc.collect()
    torch.cuda.empty_cache()

# ==========================================
# 2. Evaluation Phase (PRM Scoring)
# ==========================================
def run_evaluation(args):
    print(f"Loading PRM: {args.prm_model_path}")
    
    # PRMは普通のTransformersでロード (vLLMはRewardModel用APIが特殊なため)
    tokenizer = AutoTokenizer.from_pretrained(args.prm_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.prm_model_path,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # パス名の生成
    if "checkpoint" in args.prm_model_path:
        # models/prm_1.5b_ensemble/checkpoint-4045 -> prm_1.5b_ensemble_ckpt4045
        parent_dir = os.path.basename(os.path.dirname(args.prm_model_path))
        ckpt_name = os.path.basename(args.prm_model_path).replace("checkpoint-", "ckpt")
        prm_name = f"{parent_dir}_{ckpt_name}"
    else:
        prm_name = os.path.basename(args.prm_model_path)

    gen_model_name = os.path.basename(args.generator_name_or_path)

    for bench_name in args.target_benchmarks:
        for seed in args.seeds:
            input_file = os.path.join("generation_results", bench_name, gen_model_name, f"samples_{args.num_samples}", f"seed_{seed}.jsonl")
            if not os.path.exists(input_file):
                print(f"Input file not found, skipping: {input_file}")
                continue
            
            output_dir = os.path.join("results", bench_name, gen_model_name, prm_name, f"samples_{args.num_samples}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"seed_{seed}.jsonl")
            
            if os.path.exists(output_file) and not args.overwrite:
                print(f"Skipping scoring (exists): {output_file}")
                continue

            print(f"Scoring: {bench_name} Seed {seed} -> {output_file}")
            
            with open(input_file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            scored_results = []
            
            for item in tqdm(data, desc=f"Scoring {bench_name}"):
                problem = item["problem"]
                responses = item["responses"]
                
                batch_texts = []
                batch_indices = []

                # PRM用の入力作成
                for r_idx, resp in enumerate(responses):
                    steps = resp["steps"]
                    
                    # ★重要: 学習時との整合性チェック
                    # 学習コード:
                    #   history = [problem]
                    #   for step in steps:
                    #       history.append(step)
                    #       inputs.append("\n".join(history))
                    #
                    # ※ Chat Templateは使用しない ※
                    
                    history = [problem] # 問題文からスタート
                    
                    for s_idx, step in enumerate(steps):
                        history.append(step)
                        
                        # 学習時と同じ結合記号を使用 ("\n")
                        # Qwenの学習データ(NuminaMath)由来のフォーマットに従う
                        input_text = "\n".join(history)
                        
                        batch_texts.append(input_text)
                        batch_indices.append((r_idx, s_idx))
                    
                    # スコア格納用の枠確保
                    resp["step_scores"] = [None] * len(steps)

                # バッチ推論
                eval_batch_size = 16 
                for i in range(0, len(batch_texts), eval_batch_size):
                    batch = batch_texts[i : i + eval_batch_size]
                    indices = batch_indices[i : i + eval_batch_size]
                    
                    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(model.device)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        # Log-Sigmoidで学習した場合は、出力はLogit (スコア)
                        # 確率に戻す必要はなく、このまま大小比較に使える
                        scores = outputs.logits.squeeze(-1).cpu().tolist()
                        if isinstance(scores, float): scores = [scores]
                    
                    for (score, (r_idx, s_idx)) in zip(scores, indices):
                        responses[r_idx]["step_scores"][s_idx] = score

                scored_results.append(item)

            with open(output_file, "w", encoding="utf-8") as f:
                for r in scored_results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["generate", "evaluate", "all"])
    parser.add_argument("--target_benchmarks", nargs="+", default=["math500"], help="math500 aime24")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--generator_name_or_path", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--prm_model_path", type=str, help="Path to PRM checkpoint")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.mode in ["generate", "all"]:
        run_generation(args)
    
    if args.mode in ["evaluate", "all"]:
        if not args.prm_model_path:
            raise ValueError("Evaluate mode requires --prm_model_path")
        run_evaluation(args)