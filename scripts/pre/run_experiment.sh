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

# 環境指定
source .delta_train/bin/activate

export WANDB_API_KEY="ed1932236a317542c8c103da2af028b6ac2ee9af"
wandb login

# プロジェクト設定
YOUR_PROJECT_NAME="Delta-PRM"
YOUR_RUN_NAME="34_eval_prm_stepwise_prm_v3.0_7b_chat"

logfilename="${HOME}/log/${YOUR_RUN_NAME}.log"



# 01_生成
#python src/1_generate.py \
#    --model_id "Qwen/Qwen2.5-Math-7B-Instruct" \
#    --tensor_parallel 4 \
#    --n_paths 8


# 02_アノテーション