#!/bin/bash

# ==========================================
# 設定
# ==========================================
# 実験するシードのリスト
SEEDS=(0)

# パス設定
WORK_DIR="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM"
OUT_DIR="data/experiments/strict_beam_search_prm_1.5B_v3.0_chat_re2.0"

# モデル設定
GEN_MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
# ★重要: ここに学習済みPRMのパスを指定してください
PRM_MODEL="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM/models/prm_1.5b_30k_v3.0_chat_clean_new"

# ビームサーチ設定
BEAM_WIDTH=5
NUM_CANDIDATES=5
MAX_LOOPS=30

# ディレクトリ作成
mkdir -p "${OUT_DIR}"
mkdir -p "${WORK_DIR}/log"

# ==========================================
# ジョブ投入ループ
# ==========================================
for SEED in "${SEEDS[@]}"; do
    RUN_NAME="strict_beam_prm_1.5b_w${BEAM_WIDTH}_can${NUM_CANDIDATES}_seed${SEED}_re2.0"
    LOG_FILE="${WORK_DIR}/log/${RUN_NAME}.log"
    OUTPUT_FILE="${OUT_DIR}/seed${SEED}.json"

    echo "Submitting Job for SEED: ${SEED} ..."

    qsub <<EOF
#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=5:00:00
#PBS -P gch51650
#PBS -N ${RUN_NAME}

cd ${WORK_DIR}

# 環境設定 (ご自身の環境に合わせて調整してください)
module load cuda/12.8/12.8.1
module load cudnn/9.12/9.12.0
module load nccl/2.25/2.25.1-1
export CUDA_VISIBLE_DEVICES=0
source .delta/bin/activate

# 実行コマンド
# Pythonファイル名は main_strict_beam.py と仮定しています

python3 src/36_main_strict_beam.py \\
    --gen_model "${GEN_MODEL}" \\
    --prm_model "${PRM_MODEL}" \\
    --output_file "${OUTPUT_FILE}" \\
    --seed "${SEED}" \\
    --beam_width "${BEAM_WIDTH}" \\
    --num_candidates "${NUM_CANDIDATES}" \\
    --max_gen_loops "${MAX_LOOPS}" \\
    --gpu_memory_utilization 0.5 >& ${LOG_FILE}

EOF
    
    # ジョブスケジューラへの負荷軽減のため少し待つ
    sleep 1
done