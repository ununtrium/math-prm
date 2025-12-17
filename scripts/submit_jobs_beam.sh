#!/bin/bash

# ==========================================
# 設定エリア
# ==========================================

# 実行したいシードのリスト (必要なものをスペース区切りで記述)
# 例: "2 4 6" や "0 1 2 3 4 5 6 7 8 9" など
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# 共通設定
WIDTH="5"
CANDIDATES="5"
MEMORY="0.6"

# 出力先ディレクトリ
# ※元のコードのパス末尾が seed0_beam となっていましたが、
#   複数シードを保存する場合は汎用的な名前の方が管理しやすいかもしれません。
#   必要に応じて書き換えてください。
OUT_DIR="data/experiments/benchmark_width5_candi5_1.5b_v3.0_temp_chat_clean_new"

# 作業ディレクトリ
WORK_DIR="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM"

# ==========================================
# ループ処理
# ==========================================

for SEED in "${SEEDS[@]}"; do
    
    # ジョブ名などを定義
    RUN_NAME="32_run_beam_1.5b_30k_v3.0_width${WIDTH}_candi${CANDIDATES}_seed${SEED}_temp_chat_clean_new"
    LOG_FILE="${WORK_DIR}/log/${RUN_NAME}.log"
    
    echo "Submitting Job for SEED: ${SEED} ..."

    # qsub にスクリプト内容を直接流し込む
    # EOT内の ${変数} はこの親スクリプトの値に展開されてから qsub されます
    qsub <<EOF
#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=10:00:00
#PBS -P gch51650
#PBS -N ${RUN_NAME}

# 作業ディレクトリ
cd ${WORK_DIR}

# モジュールロード
module load cuda/12.8/12.8.1
module load cudnn/9.12/9.12.0
module load nccl/2.25/2.25.1-1

# GPU 指定
export CUDA_VISIBLE_DEVICES=0

# 環境指定
source .delta/bin/activate

# 実行コマンド
# (変数はすでに親スクリプトで展開された値が入ります)
python3 src/32_run_benchmark_flexible_1.5b.py \\
    --seeds "${SEED}" \\
    --beam_width "${WIDTH}" \\
    --num_candidates "${CANDIDATES}" \\
    --gpu_memory_utilization "${MEMORY}" \\
    --output_dir "${OUT_DIR}" \\
    --num_gpus 1 >& ${LOG_FILE}

EOF

    # スケジューラへの負荷軽減のため少し待機
    sleep 1

done

echo "All jobs submitted."