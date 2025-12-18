#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=6:00:00
#PBS -P gch51650


# 作業ディレクトリ
HOME="/groups/gch51650/kawahara_lab/enomoto/self-correct/Delta-PRM"
MODEL_DIR="/groups/gch51650/kawahara_lab/enomoto/self-correct/download_models"
cd ${HOME}

# モジュールロード
module load cuda/12.1/12.1.1
module load cudnn/9.12/9.12.0
module load nccl/2.25/2.25.1-1

# GPU 指定
export CUDA_VISIBLE_DEVICES=0

# プロジェクト設定
YOUR_PROJECT_NAME="Delta-PRM"
YOUR_RUN_NAME="34_eval_prm_stepwise_prm_v3.0_7b_chat"

# 環境指定
source .delta_train/bin/activate

export WANDB_API_KEY="ed1932236a317542c8c103da2af028b6ac2ee9af"
wandb login


logfilename="${HOME}/log/${YOUR_RUN_NAME}.log"


python3 src/34_eval_prm_stepwise.py \
    --model_path "models/prm_7b_30k_v3.0_chat_clean_new" \
    --input_path "data/experiments/majority_voting_n16" \
    --output_dir "data/experiments/bon_stepwise_scored_prm_v3.0_7b_chat" \
    --batch_size 8 >& ${logfilename}