#!/bin/bash

# ========================================================
# 評価したいPRMモデルのリスト
# ========================================================
MODELS=(
    "models/prm_1.5b_ensemble_raw_only/checkpoint-4054"
    "models/prm_1.5b_qwen_raw_only/checkpoint-4054"
    "models/orm_1.5b_raw_only/checkpoint-4054"
    "models/prm_1.5b_deepseek_raw_only/checkpoint-4054"
    "models/prm_1.5b_llama_raw_only/checkpoint-4054"
    "models/prm_1.5b_30k_v3.0_chat_clean_new/checkpoint-8087"
)

# 実行パラメータ
BEAM_WIDTH=20
NUM_CANDIDATES=10
SEEDS="1 2 3 4"
WORKER_SCRIPT="scripts/42_evaluate_tree_search.sh"

# ========================================================
# ループ処理でジョブ投入
# ========================================================
for MODEL_PATH in "${MODELS[@]}"; do
    if [ -d "$MODEL_PATH" ]; then
        MODEL_DIR_NAME=$(basename $(dirname "$MODEL_PATH"))
        
        # ジョブ名
        JOB_NAME="tree_${MODEL_DIR_NAME}"
        LOG_NAME="${JOB_NAME}"
        
        echo "Submitting: $JOB_NAME (Seeds: $SEEDS)"
        
        qsub -N "$JOB_NAME" \
             -v PRM_MODEL_PATH="$MODEL_PATH",LOG_NAME="$LOG_NAME",BEAM_WIDTH=$BEAM_WIDTH,NUM_CANDIDATES=$NUM_CANDIDATES,SEEDS="$SEEDS" \
             "$WORKER_SCRIPT"
             
    else
        echo "WARNING: Model path not found: $MODEL_PATH"
    fi
done

echo "All jobs submitted!"