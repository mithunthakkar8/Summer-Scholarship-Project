#!/bin/bash -e
#SBATCH --job-name=basic_great_sem
#SBATCH --cpus-per-task=8      
#SBATCH --gres=gpu:1 
#SBATCH --mem=32G                 
#SBATCH --time=3:00:00           
module --force purge
module load NeSI/zen3

source /nesi/project/vuw04485/pisa_env/bin/activate

export PYTHONPATH=/nesi/project/vuw04485/third_party/Tabula-main:$PYTHONPATH
export HF_HOME=/nesi/project/vuw04485/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/pisa_env/bin/activate

cd /nesi/project/vuw04485/pisa_sem_pipeline/python
python 006_python_basic_tabula_pipeline_distilgpt2.py

