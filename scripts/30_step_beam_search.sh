#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=8:00:00
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

# プロジェクト設定
YOUR_PROJECT_NAME="Delta-PRM"
YOUR_RUN_NAME="30_step_beam_search_7b_30k_v3.0"

# 環境指定
source .delta/bin/activate

#export WANDB_API_KEY="ed1932236a317542c8c103da2af028b6ac2ee9af"
#wandb login


logfilename="${HOME}/log/${YOUR_RUN_NAME}.log"

python3 src/30_step_beam_search.py \
    --prm_model models/prm_7b_30k_v3.0/checkpoint-16174 \
    --beam_width 3 \
    --output_file data/math500_results_beam_7b_30k_v3.0.json >& ${logfilename}