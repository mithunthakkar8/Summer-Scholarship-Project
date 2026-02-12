#!/bin/bash -e
#SBATCH --job-name=basic_predllm_sem
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/pisa_env/bin/activate

# ---- GPU safety
export CUDA_VISIBLE_DEVICES=0

# ---- Pred-LLM visibility
export PYTHONPATH=/nesi/project/vuw04485/Pred-LLM:$PYTHONPATH
export PYTHONPATH=/nesi/project/vuw04485/Pred-LLM:/nesi/project/vuw04485/Tabula:$PYTHONPATH


# ---- HuggingFace cache (project-local)
export HF_HOME=/nesi/project/vuw04485/hf_cache
export TRANSFORMERS_CACHE=/nesi/project/vuw04485/hf_cache
export HF_HUB_CACHE=/nesi/project/vuw04485/hf_cache

# ---- Tokenizer noise suppression
export TOKENIZERS_PARALLELISM=false

cd /nesi/project/vuw04485/pisa_sem_pipeline/python
python 007_python_basic_predllm_great_LV_distilgpt2.py
