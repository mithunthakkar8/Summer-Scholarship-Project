#!/bin/bash -e
#SBATCH --job-name=great_sem
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8         
#SBATCH --mem=32G                 
#SBATCH --time=8:00:00           

module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/pisa_env/bin/activate

# --------------------------------------------------
# Hugging Face + temp cache redirection (CRITICAL)
# --------------------------------------------------
export HF_HOME=/nesi/project/vuw04485/hf_cache
export TRANSFORMERS_CACHE=/nesi/project/vuw04485/hf_cache
export HF_DATASETS_CACHE=/nesi/project/vuw04485/hf_cache
export HF_XET_CACHE=/nesi/project/vuw04485/hf_xet
export TMPDIR=/nesi/project/vuw04485/tmp

mkdir -p /nesi/project/vuw04485/hf_cache
mkdir -p /nesi/project/vuw04485/hf_xet
mkdir -p /nesi/project/vuw04485/tmp
mkdir -p logs

cd /nesi/project/vuw04485/pisa_sem_pipeline/python
python 004_python_multiphase_great_pipeline_C.py

