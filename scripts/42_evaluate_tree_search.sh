#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gch51650

# 注意: -N (ジョブ名) は qsub コマンド側で指定

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
# パラメータ (qsubの -v オプションから受け取る)
# ========================================================
# 必須: PRM_MODEL_PATH, LOG_NAME
# 任意: BEAM_WIDTH, NUM_CANDIDATES, SEED, BENCHMARKS

# デフォルト値
GEN_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
BENCHMARKS=${BENCHMARKS:-"math500 aime24 aime25"}
BEAM_WIDTH=${BEAM_WIDTH:-5}
NUM_CANDIDATES=${NUM_CANDIDATES:-5}
SEEDS=${SEEDS:-"0 1 2"}

# ログディレクトリ
mkdir -p "${HOME_DIR}/log/tree_search"

# ========================================================
# 実行処理
# ========================================================
echo "Starting Tree Search..."
echo "Generator: $GEN_MODEL"
echo "PRM: $PRM_MODEL_PATH"
echo "Params: Beam=$BEAM_WIDTH, Cand=$NUM_CANDIDATES, Seeds=$SEEDS"

SCRIPT_PATH="src/42_evaluate_tree_search.py"

python3 ${SCRIPT_PATH} \
    --gen_model "$GEN_MODEL" \
    --prm_model "$PRM_MODEL_PATH" \
    --target_benchmarks $BENCHMARKS \
    --beam_width $BEAM_WIDTH \
    --num_candidates $NUM_CANDIDATES \
    --seeds $SEEDS \
    --gpu_memory_utilization 0.6 \
    >& "${HOME_DIR}/log/tree_search/${LOG_NAME}.log"

echo "Job Finished!"