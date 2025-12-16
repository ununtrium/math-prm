#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -P gch51650


# 作業ディレクトリ
HOME="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM"
MODEL_DIR="/groups/gch51650/kawahara_lab/enomoto/self-correct/download_models"
cd ${HOME}

# モジュールロード
module load cuda/12.8/12.8.1
module load cudnn/9.12/9.12.0
module load nccl/2.25/2.25.1-1

# GPU 指定
export CUDA_VISIBLE_DEVICES=0

# 環境指定
source .delta/bin/activate

#export WANDB_API_KEY="ed1932236a317542c8c103da2af028b6ac2ee9af"
#wandb login


TARGET_SEED="4"  # 次は "4", その次は "6" に書き換えてqsub
WIDTH="20"
CANDIDATES="10"
MEMORY="0.6"
OUT_DIR="data/experiments/benchmark_width20_candi10_seed0_beam"

# プロジェクト設定
YOUR_PROJECT_NAME="Delta-PRM"
YOUR_RUN_NAME="32_run_beam_7b_30k_v3.0_width${WIDTH}_candi${CANDIDATES}_seed${TARGET_SEED}"

logfilename="${HOME}/log/${YOUR_RUN_NAME}.log"

python3 src/32_run_benchmark_flexible.py \
    --seeds "${TARGET_SEED}" \
    --beam_width "${WIDTH}" \
    --num_candidates "${CANDIDATES}" \
    --gpu_memory_utilization "${MEMORY}" \
    --output_dir "${OUT_DIR}" \
    --num_gpus 1 >& ${logfilename}