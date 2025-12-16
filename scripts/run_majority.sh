#!/bin/bash

# 設定
SEEDS=(0 1 2 3 4 5 6 7 8 9)
N_SAMPLES=16
MODEL="Qwen/Qwen2.5-Math-7B-Instruct"
WORK_DIR="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM"
OUT_DIR="data/experiments/majority_voting_n${N_SAMPLES}"

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="majority_voting_7b_n${N_SAMPLES}_seed${SEED}"
    LOG_FILE="${WORK_DIR}/log/${RUN_NAME}.log"
    OUTPUT_FILE="${OUT_DIR}/seed${SEED}.json"

    echo "Submitting Job for SEED: ${SEED} ..."

    qsub <<EOF
#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=3:00:00
#PBS -P gch51650
#PBS -N ${RUN_NAME}

cd ${WORK_DIR}
module load cuda/12.8/12.8.1
module load cudnn/9.12/9.12.0
module load nccl/2.25/2.25.1-1
export CUDA_VISIBLE_DEVICES=0
source .delta/bin/activate

python3 src/33_run_majority_voting_and_save.py \\
    --model "${MODEL}" \\
    --output_file "${OUTPUT_FILE}" \\
    --seed "${SEED}" \\
    --n "${N_SAMPLES}" \\
    --temperature 0.7 \\
    --gpu_memory_utilization 0.8 >& ${LOG_FILE}
EOF
    sleep 1
done