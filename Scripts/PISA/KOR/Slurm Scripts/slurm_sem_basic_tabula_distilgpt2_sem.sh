#!/bin/bash -e
#SBATCH --job-name=basic_tabula_sem_gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GA100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8        
#SBATCH --mem=32G 



# ==============================
# Hugging Face cache (SCRATCH)
# ==============================
export PYTHONPATH=/nfs/home/thakkamith/projects/TIMSS/third_party/Tabula:$PYTHONPATH

export HF_HOME=/nfs/scratch/thakkamith/hf_cache
export HF_HUB_CACHE=/nfs/scratch/thakkamith/hf_cache

# Optional: ONLY enable offline if models are already cached
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

module purge
module load GCCcore/12.2.0
module load Python/3.10.8

source /nfs/home/thakkamith/venvs/sem_venv/bin/activate

cd /nfs/home/thakkamith/projects/PISA/scripts
python 006_python_basic_tabula_pipeline_distilgpt2_sem.py

