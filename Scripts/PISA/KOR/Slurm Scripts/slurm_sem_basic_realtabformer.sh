#!/bin/bash -e
#SBATCH --job-name=basic_realtabformer
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00

module purge

# 1) Toolchain FIRST
module load NeSI/zen3

# 2) CUDA SECOND
module load CUDA/12.1.1

# 3) Python THIRD
module load Python/3.10.5-gimkl-2022a

# 4) Virtual environment
source /nesi/project/vuw04485/pisa_env/bin/activate

# Environment hygiene
export PYTHONPATH=/nesi/project/vuw04485/Pred-LLM:$PYTHONPATH
export HF_HOME=/nesi/project/vuw04485/hf_cache

cd /nesi/project/vuw04485/pisa_sem_pipeline/python

# ---- HARD FAIL IF CUDA IS NOT VISIBLE ----
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not visible — aborting to avoid CPU fallback")
EOF

# ---- Actual run ----
python 008_python_basic_realtabformer.py
