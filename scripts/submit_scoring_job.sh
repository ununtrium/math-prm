#!/bin/bash

# ========================================================
# 評価したいPRMモデルのリスト
# ========================================================
# パスは実際のディレクトリ構造に合わせてください
MODELS=(
    "models/prm_1.5b_ensemble_raw_only_no_trigger/checkpoint-4054"
    "models/prm_1.5b_qwen_raw_only_no_trigger/checkpoint-4054"
    #"models/orm_1.5b_raw_only_no_trigger/checkpoint-4054"
    # 必要に応じて追加 (Llama, DeepSeekなど)
    "models/prm_1.5b_deepseek_raw_only_no_trigger/checkpoint-4054"
    "models/prm_1.5b_llama_raw_only_no_trigger/checkpoint-4054"
)

# 実行するPBSスクリプト
WORKER_SCRIPT="scripts/40_evaluate_best_of_n_score.sh"

# ========================================================
# ループ処理でジョブ投入
# ========================================================
for MODEL_PATH in "${MODELS[@]}"; do
    if [ -d "$MODEL_PATH" ]; then
        # モデル名を見やすく整形 (例: prm_1.5b_ensemble)
        # ディレクトリ名から抽出 (親ディレクトリ名を取得)
        MODEL_DIR_NAME=$(basename $(dirname "$MODEL_PATH"))
        
        # ログファイル名とジョブ名を設定
        JOB_NAME="score_${MODEL_DIR_NAME}"
        LOG_NAME="${JOB_NAME}_numina"
        
        echo "Submitting job for: $MODEL_DIR_NAME"
        
        # qsub コマンド
        # -N: ジョブ名
        # -v: 環境変数としてスクリプトに値を渡す
        qsub -N "$JOB_NAME" \
             -v PRM_MODEL_PATH="$MODEL_PATH",LOG_NAME="$LOG_NAME" \
             "$WORKER_SCRIPT"
             
    else
        echo "WARNING: Model path not found, skipping: $MODEL_PATH"
    fi
done

echo "All scoring jobs submitted!"