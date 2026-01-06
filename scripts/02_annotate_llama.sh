#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=14:00:00
#PBS -P gch51650


# 作業ディレクトリ
HOME="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM"
cd ${HOME}

# モジュールロード
module load cuda/12.8/12.8.1
module load cudnn/9.12/9.12.0
module load nccl/2.25/2.25.1-1

# GPU 指定
export CUDA_VISIBLE_DEVICES=0

# 環境指定
source .delta_train/bin/activate

# 1. アノテーションに使用するモデル
# PRMとして学習させる予定のモデル(1.5B)を指定することを強く推奨します。
MODEL_ID="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM/models/Llama-3.1-8B-Instruct" # ここを変更
MODEL_NAME=$(basename "${MODEL_ID}")

# 2. 入力ファイル (生成フェーズで作成したファイル)
INPUT_FILE="data/numinamath_gen_15k.jsonl"

# 3. 出力ファイル (アノテーション結果)
OUTPUT_FILE="data/02_annotate/${MODEL_NAME}/prm_annotated_1M_no_trigger.jsonl"

# 4. バッチサイズ
# 1.5BモデルならVRAMに余裕があるため、16~32でも動く可能性があります。
# OOM(Out of Memory)が出る場合は 8 や 4 に下げてください。
BATCH_SIZE=8

# プロジェクト設定
YOUR_PROJECT_NAME="Delta-PRM"
YOUR_RUN_NAME="02_annotate_${MODEL_NAME}_prm_1M_no_trigger"


logfilename="${HOME}/log/${YOUR_RUN_NAME}.log"

mkdir -p "$(dirname "${OUTPUT_FILE}")"


# ========================================================
# 実行処理
# ========================================================

echo "--------------------------------------------------"
echo "Starting PRM Annotation Pipeline"
echo "--------------------------------------------------"
echo "Model ID    : $MODEL_ID"
echo "Input File  : $INPUT_FILE"
echo "Output File : $OUTPUT_FILE"
echo "Batch Size  : $BATCH_SIZE"
echo "--------------------------------------------------"

# Pythonパスの設定 (srcモジュールを読み込めるようにする)

# アノテーション実行
python3 src/20_annotate.py \
    --model_id "$MODEL_ID" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --batch_size $BATCH_SIZE >& ${logfilename}

echo ""
echo "--------------------------------------------------"
echo "Annotation Completed Successfully!"
echo "Data saved to: $OUTPUT_FILE"
echo "--------------------------------------------------"