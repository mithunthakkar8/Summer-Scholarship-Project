#!/bin/bash -e
#SBATCH --job-name=basic_tabby_sem
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/pisa_env/bin/activate

export PYTHONPATH=/nesi/project/vuw04485/third_party/tabby:$PYTHONPATH

# ---- HF cache consistency (strongly recommended)
export HF_HOME=/nesi/project/vuw04485/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME


cd /nesi/project/vuw04485/pisa_sem_pipeline/python
python 008_python_basic_tabby_distilgpt2.py
