#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -P gch51650

set -e

# ========================================================
# 環境設定
# ========================================================
HOME_DIR="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM"
cd ${HOME_DIR}

export WANDB_API_KEY="ed1932236a317542c8c103da2af028b6ac2ee9af"
wandb login

module load cuda/12.8/12.8.1
module load cudnn/9.12/9.12.0
module load nccl/2.25/2.25.1-1
export CUDA_VISIBLE_DEVICES=0
source .delta_train/bin/activate

# ========================================================
# パラメータ設定
# ========================================================

BASE_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
TRAIN_FILE="data/02_annotate/deepseek-math-7b-instruct/prm_annotated_1M_no_trigger.jsonl"
OUTPUT_DIR="models/prm_1.5b_deepseek_raw_only_no_trigger"  # 名前をわかりやすく変更
RUN_NAME="prm_1.5b_deepseek_raw_only_no_trigger"

# ハイパーパラメータ
LEARNING_RATE=2e-5
BATCH_SIZE=8
GRAD_ACCUM=32  # 実質Batch=256
EPOCHS=1
MAX_LEN=3072

# ========================================================
# 実行処理
# ========================================================
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${HOME_DIR}/log"

echo "Starting PRM Training (Raw Prob Only)..."
echo "Model: $BASE_MODEL"
echo "Data:  $TRAIN_FILE"

# ... (前略)
# 修正箇所: パラメータ指定
# tau: -5.0 (分布の中央付近)
# alpha: 1.0 (標準的な傾き)
# aux_weight: 0.0 (Delta不使用)

python3 src/30_train.py \
    --train_file "$TRAIN_FILE" \
    --base_model "$BASE_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --wandb_project "Delta-PRM" \
    --run_name "$RUN_NAME" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_epochs "$EPOCHS" \
    --max_length "$MAX_LEN" \
    --test_size 2000 \
    --alpha 1.0 \
    --tau -5.0 \
    --aux_weight 0.0 \
    --beta 0.0 \
    --seed 42 >& "${HOME_DIR}/log/train_${RUN_NAME}.log"

echo "Training Finished!"