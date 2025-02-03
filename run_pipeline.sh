#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=LLMRubric
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --account=a100acct
#SBATCH --partition=gpu-a100
#SBATCH --gpus=1
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/12.1

conda activate llm_rubric_env

python /export/fs06/psingh54/LLMRubric/main.py --dataset /export/fs06/psingh54/LLMRubric/data/hanna.yaml --output /export/fs06/psingh54/LLMRubric/outputs