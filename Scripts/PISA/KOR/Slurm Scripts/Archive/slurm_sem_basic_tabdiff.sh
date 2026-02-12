#!/bin/bash -e
#SBATCH --job-name=tabdiff_pisa
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=/nesi/project/vuw04485/slurm/tabdiff_%j.out
#SBATCH --error=/nesi/project/vuw04485/slurm/tabdiff_%j.err

# -------------------------------
# Clean environment
# -------------------------------
module --force purge
unset PYTHONPATH
unset PYTHONHOME

module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

# -------------------------------
# Activate pip virtualenv
# -------------------------------
source /nesi/project/vuw04485/venvs/tabdiff/bin/activate

# -------------------------------
# Redirect caches off HOME
# -------------------------------
export HF_HOME=/nesi/project/vuw04485/hf_cache
export TRANSFORMERS_CACHE=/nesi/project/vuw04485/hf_cache
export TORCH_HOME=/nesi/project/vuw04485/torch_cache
export XDG_CACHE_HOME=/nesi/project/vuw04485/.cache

mkdir -p /nesi/project/vuw04485/{hf_cache,torch_cache,.cache}

# -------------------------------
# Ensure TabDiff is importable
# -------------------------------
export PYTHONPATH=/nesi/project/vuw04485/third_party/TabDiff:$PYTHONPATH

cd /nesi/project/vuw04485/third_party/TabDiff

# -------------------------------
# Hard sanity check (FAIL FAST)
# -------------------------------
python - << 'EOF'
import sys, torch
print("Python executable:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
import icecream
print("Sanity check passed")
EOF

# -------------------------------
# Train
# -------------------------------
python main.py \
  --dataname pisa_sgp \
  --mode train \
  --gpu 0 \
  --no_wandb

# -------------------------------
# Sample
# -------------------------------
python main.py \
  --dataname pisa_sgp \
  --mode test \
  --num_samples_to_generate 6253 \
  --num_runs 5 \
  --report \
  --no_wandb
