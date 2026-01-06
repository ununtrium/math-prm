#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=6:00:00
#PBS -P gch51650

# 注意: -N (ジョブ名) は qsub コマンド側で指定します

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
# パラメータ設定 (qsubの -v オプションから受け取る)
# ========================================================
# PRM_MODEL_PATH: 評価するモデルのパス
# LOG_NAME: ログファイルの名前

GENERATOR_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
BENCHMARKS="numina_train"
SEEDS="42"
NUM_SAMPLES="16"

# ログディレクトリ
mkdir -p "${HOME_DIR}/log/scoring"

# ========================================================
# 実行処理 (Evaluate Mode)
# ========================================================
echo "Starting PRM Evaluation..."
echo "Generator: $GENERATOR_MODEL"
echo "PRM Model: $PRM_MODEL_PATH"

SCRIPT_PATH="src/40_evaluate_best_of_n.py"

# python実行
python3 ${SCRIPT_PATH} \
    --mode evaluate \
    --generator_name_or_path "$GENERATOR_MODEL" \
    --prm_model_path "$PRM_MODEL_PATH" \
    --target_benchmarks $BENCHMARKS \
    --seeds $SEEDS \
    --num_samples $NUM_SAMPLES \
    >& "${HOME_DIR}/log/scoring/${LOG_NAME}.log"

echo "Scoring Finished for $PRM_MODEL_PATH"