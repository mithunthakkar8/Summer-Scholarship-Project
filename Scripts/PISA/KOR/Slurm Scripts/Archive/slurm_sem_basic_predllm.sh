#!/bin/bash -e
#SBATCH --job-name=basic_great_sem_gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GA100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8        
#SBATCH --mem=32G                 
  

module purge
module load GCCcore/12.2.0
module load Python/3.10.8

source /nfs/home/thakkamith/venvs/sem_venv/bin/activate

# HuggingFace cache (CRITICAL)
export HF_HOME=/nfs/scratch/thakkamith/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

# Pred-LLM path
export PYTHONPATH=/nfs/home/thakkamith/projects/TIMSS/third_party/Pred-LLM:$PYTHONPATH

cd /nfs/home/thakkamith/projects/TIMSS/third_party/Pred-LLM

# ----------------------------
# Run Pred-LLM (Regression)
# ----------------------------
python -W ignore pred_llm_reg.py \
  --dataset pisa \
  --data_path /nfs/scratch/thakkamith/PISA/data/df_core_with_smp_latent_SGP.csv \
  --target LV_SMP \
  --method pred_llm \
  --trainsize 0.8 \
  --testsize 0.2 \
  --runs 5 \
  --output_dir /nfs/scratch/thakkamith/PISA/outputs/predllm