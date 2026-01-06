import os
import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
import utils  # utils.py

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Error: vllm not found. Please install via: pip install vllm")
    exit()

# ==========================================
# ベンチマーク設定
# ==========================================
BENCHMARK_CONFIG = {
    "math500": {"path": "HuggingFaceH4/math-500", "split": "test", "key_prob": "problem", "key_ans": "answer"},
    "aime24": {"path": "HuggingFaceH4/aime_2024", "split": "train", "key_prob": "problem", "key_ans": "answer"},
    "aime25": {"path": "math-ai/aime25", "split": "test", "key_prob": "problem", "key_ans": "answer"},
}

def load_benchmark_data(bench_name):
    cfg = BENCHMARK_CONFIG.get(bench_name)
    if not cfg:
        if os.path.exists(bench_name):
            return load_dataset("json", data_files=bench_name, split="train")
        raise ValueError(f"Unknown benchmark: {bench_name}")
    
    print(f"Loading benchmark: {bench_name}")
    if cfg.get("format") == "json":
        dataset = load_dataset("json", data_files=cfg["path"], split="train")
    else:
        try:
            dataset = load_dataset(cfg["path"], split=cfg.get("split", "test"))
        except:
            dataset = load_dataset(cfg["path"], split="train")
    
    standardized = []
    for item in dataset:
        prob = item.get(cfg["key_prob"])
        ans = item.get(cfg["key_ans"])
        if ans is None and "solution" in item:
            ans = item["solution"]
        standardized.append({"problem": prob, "gold_answer": ans})
    return standardized

# ==========================================
# 設定
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--prm_model", type=str, required=True)
    parser.add_argument("--target_benchmarks", nargs="+", default=["math500"], help="math500 aime24")
    
    # 実験パラメータ
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="List of seeds to run")
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--num_candidates", type=int, default=5) 
    
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()

# ==========================================
# ビームサーチクラス
# ==========================================
class StepBeamSearcher:
    def __init__(self, llm_engine, prm_model, prm_tokenizer, device):
        self.llm = llm_engine
        self.prm_model = prm_model
        self.prm_tokenizer = prm_tokenizer
        self.device = device

    def get_prm_scores(self, problems, steps_lists):
        batch_texts = []
        for prob, steps in zip(problems, steps_lists):
            full_text = "\n".join([prob] + steps)
            batch_texts.append(full_text)
        
        scores = []
        batch_size = 8
        for i in range(0, len(batch_texts), batch_size):
            batch = batch_texts[i : i + batch_size]
            inputs = self.prm_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=3072
            ).to(self.device)
            
            with torch.no_grad():
                out = self.prm_model(**inputs)
            scores.extend(out.logits.squeeze(-1).tolist())
            
        return scores

    def search(self, problem_text, gen_tokenizer_prompt, seed, beam_width=5, max_steps=20, num_candidates=5):
        beam = [{
            "gen_text": gen_tokenizer_prompt,
            "prm_steps": [],
            "score": 0.0,
            "finished": False,
            "id": 0 # ID管理用
        }]
        
        dropped_finished_pool = []
        termination_reason = "max_steps_reached"
        
        # ★履歴保存用リスト
        # 各ステップごとの候補一覧を保存する
        history_log = [] 
        
        node_counter = 1 # ユニークID用

        sampling_params = SamplingParams(
            n=num_candidates, 
            temperature=0.7, 
            top_p=0.95, 
            max_tokens=512,
            stop=["\n\n", "<|im_end|>"], 
            seed=seed
        )

        for t in range(max_steps):
            # 1. Expand
            prompts = []
            metadata = [] 
            finished_in_beam = []

            for node in beam:
                if node["finished"] or len(node["gen_text"]) > 12000:
                    finished_in_beam.append(node)
                    continue
                prompts.append(node["gen_text"])
                metadata.append(node)
            
            if not prompts:
                termination_reason = "no_prompts_to_expand"
                break

            try:
                outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
            except ValueError:
                termination_reason = "generation_error"
                break

            # 2. Parse
            next_candidates_prm_input_prob = []
            next_candidates_prm_input_steps = [] 
            next_candidates_objs = [] 
            
            for i, request_output in enumerate(outputs):
                parent_node = metadata[i]
                
                for completion in request_output.outputs:
                    step_text = completion.text
                    
                    if "\n\n" in step_text: step_text = step_text.split("\n\n")[0]
                    step_text = step_text.replace("<|im_end|>", "").strip()
                    if not step_text: continue
                    
                    new_gen_text = parent_node["gen_text"] + "\n" + step_text
                    new_prm_steps = parent_node["prm_steps"] + [step_text]
                    is_finished = (utils.extract_answer_content(step_text) is not None)
                    
                    next_candidates_prm_input_prob.append(problem_text)
                    next_candidates_prm_input_steps.append(new_prm_steps)
                    
                    next_candidates_objs.append({
                        "gen_text": new_gen_text,
                        "prm_steps": new_prm_steps,
                        "finished": is_finished,
                        "parent_id": parent_node["id"], # 親のIDを記録
                        "step_text": step_text # ログ用にテキスト保持
                    })

            if not next_candidates_objs:
                termination_reason = "no_valid_candidates_generated"
                break

            # 3. Score
            step_scores = self.get_prm_scores(next_candidates_prm_input_prob, next_candidates_prm_input_steps)

            # 4. Update & Prune
            candidates_pool = []
            for obj, step_score in zip(next_candidates_objs, step_scores):
                new_score = step_score
                candidates_pool.append({
                    "id": node_counter, # 新規ID
                    "gen_text": obj["gen_text"],
                    "prm_steps": obj["prm_steps"],
                    "score": new_score,
                    "finished": obj["finished"],
                    "parent_id": obj["parent_id"],
                    "step_text": obj["step_text"]
                })
                node_counter += 1
            
            # 完了済みノードも混ぜる
            candidates_pool.extend(finished_in_beam)
            
            # ソート
            candidates_pool.sort(key=lambda x: x["score"], reverse=True)
            
            # 次のビーム
            next_beam = candidates_pool[:beam_width]
            
            # 脱落パスの救出
            dropped = candidates_pool[beam_width:]
            for d_node in dropped:
                if d_node["finished"]:
                    dropped_finished_pool.append(d_node)
            
            # ★履歴の記録 (このステップで生成・評価された全候補)
            step_log = []
            next_beam_ids = set(n["id"] for n in next_beam) # 高速検索用
            
            for cand in candidates_pool:
                # 完了済みでただ引き継がれただけのノードはログに残すと重複するのでスキップするか、
                # あるいは "status": "kept" として残すか。ここでは新規生成分だけ見るために step_text があるものに限定してもよいが、
                # 全容把握のため全記録する。
                
                status = "pruned"
                if cand["id"] in next_beam_ids:
                    status = "selected"
                elif cand in dropped_finished_pool:
                    status = "dropped_saved"
                
                step_log.append({
                    "step_idx": t + 1,
                    "id": cand["id"],
                    "parent_id": cand.get("parent_id"), # 初期の完了済みノードはキーがない場合があるのでget
                    "score": cand["score"],
                    "step_text": cand.get("step_text", "(finished_node)"),
                    "status": status,
                    "finished": cand["finished"]
                })
            
            history_log.append(step_log)
            
            beam = next_beam
            
            if all(n["finished"] for n in beam):
                termination_reason = "all_beam_finished"
                break
        
        # ==========================================
        # 最終選択
        # ==========================================
        beam_finished = [n for n in beam if n["finished"]]
        if beam_finished:
            best_node = max(beam_finished, key=lambda x: x["score"])
            return best_node, "beam_finished", termination_reason, history_log
        elif dropped_finished_pool:
            best_node = max(dropped_finished_pool, key=lambda x: x["score"])
            return best_node, "dropped_finished", termination_reason, history_log
        else:
            return beam[0], "unfinished", termination_reason, history_log

# ==========================================
# Main
# ==========================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading PRM: {args.prm_model}")
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_model)
    prm_model = AutoModelForSequenceClassification.from_pretrained(
        args.prm_model, num_labels=1, torch_dtype=torch.bfloat16, device_map=device
    )
    prm_model.eval()
    
    print(f"Loading Generator: {args.gen_model}")
    llm = LLM(
        model=args.gen_model, 
        tensor_parallel_size=1, 
        gpu_memory_utilization=args.gpu_memory_utilization, 
        dtype="bfloat16", 
        enforce_eager=True,
        max_model_len=args.max_model_len
    )
    gen_tokenizer = AutoTokenizer.from_pretrained(args.gen_model)

    if "checkpoint" in args.prm_model:
        p_dir = os.path.basename(os.path.dirname(args.prm_model))
        c_name = os.path.basename(args.prm_model).replace("checkpoint-", "ckpt")
        prm_name = f"{p_dir}_{c_name}"
    else:
        prm_name = os.path.basename(args.prm_model)

    gen_name = os.path.basename(args.gen_model)

    searcher = StepBeamSearcher(llm, prm_model, prm_tokenizer, device)
    
    for bench_name in args.target_benchmarks:
        print(f"Starting Benchmark: {bench_name}")
        output_dir = os.path.join("results", "tree_search", bench_name, gen_name, prm_name)
        os.makedirs(output_dir, exist_ok=True)
        
        dataset = load_benchmark_data(bench_name)

        for seed in args.seeds:
            set_seed(seed)
            print(f"  >> Running Seed: {seed}")
            filename = f"beam{args.beam_width}_cand{args.num_candidates}_seed{seed}.json"
            output_file = os.path.join(output_dir, filename)
            if os.path.exists(output_file) and not args.overwrite:
                print(f"     Skipping (exists): {output_file}")
                continue

            results = []
            correct_count = 0
            
            for item in tqdm(dataset, mininterval=30, desc=f"Searching {bench_name} (Seed {seed})"):
                problem = item["problem"]
                gold_content = utils.extract_answer_content(item["gold_answer"])
                if gold_content is None: gold_content = item["gold_answer"]

                if "Instruct" in args.gen_model:
                    gen_prompt = utils.build_prompt(gen_tokenizer, problem)
                else:
                    gen_prompt = (
                        f"Problem:\n{problem}\n\n"
                        "Please reason step by step and put your final answer within \\boxed{}.\n\n"
                        "Solution:"
                    )
                
                # ★変更: history_log を受け取る
                best_node, select_source, term_reason, history_log = searcher.search(
                    problem_text=problem,
                    gen_tokenizer_prompt=gen_prompt,
                    seed=seed,
                    beam_width=args.beam_width,
                    max_steps=args.max_steps,
                    num_candidates=args.num_candidates
                )
                
                full_solution_text = "\n".join(best_node["prm_steps"])
                pred_content = utils.extract_answer_content(full_solution_text)
                is_correct = utils.check_equivalence(pred_content, gold_content)
                
                if is_correct: correct_count += 1
                
                results.append({
                    "problem": problem,
                    "gold": gold_content,
                    "pred": pred_content,
                    "is_correct": is_correct,
                    "score": best_node["score"],
                    "select_source": select_source,
                    "termination_reason": term_reason,
                    "steps": best_node["prm_steps"],
                    "tree_history": history_log # ★ここに全履歴を保存
                })
                
            acc = correct_count / len(dataset) * 100
            print(f"     Seed {seed} Finished. Accuracy: {acc:.2f}%")
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "benchmark": bench_name,
                    "model": args.prm_model,
                    "accuracy": acc,
                    "config": {
                        "beam_width": args.beam_width, 
                        "num_candidates": args.num_candidates,
                        "seed": seed
                    },
                    "details": results
                }, f, indent=2, ensure_ascii=False)
            print(f"     Saved results to {output_file}")
if __name__ == "__main__":
    main()