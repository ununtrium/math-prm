#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=2:00:00
#PBS -P gch51650
#PBS -N gen_qwen_1.5b_base

set -e

# ========================================================
# 環境設定
# ========================================================
HOME_DIR="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM"
cd ${HOME_DIR}

module load cuda/12.8/12.8.1
module load cudnn/9.12/9.12.0
module load nccl/2.25/2.25.1-1
export CUDA_VISIBLE_DEVICES=0
source .delta/bin/activate

# ========================================================
# パラメータ設定
# ========================================================
GENERATOR_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
BENCHMARKS="aime25"  # 対象ベンチマーク
SEEDS="0 1 2"                       # シード
NUM_SAMPLES=64                      # サンプル数

# ログディレクトリ作成
mkdir -p "${HOME_DIR}/log"

# ========================================================
# 実行処理 (Generate Mode)
# ========================================================
echo "Starting Generation with vLLM..."
echo "Model: $GENERATOR_MODEL"

# evaluate_best_of_n.py のパスは適宜調整してください (例: src/40_evaluate_best_of_n.py)
SCRIPT_PATH="src/40_evaluate_best_of_n.py"

python3 ${SCRIPT_PATH} \
    --mode generate \
    --generator_name_or_path "$GENERATOR_MODEL" \
    --target_benchmarks $BENCHMARKS \
    --seeds $SEEDS \
    --num_samples $NUM_SAMPLES \
    >& "${HOME_DIR}/log/gen_qwen_1.5b_base.log"

echo "Generation Finished!"