#!/bin/bash -e
#SBATCH --job-name=tabdiff_train
#SBATCH --partition=genoa
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/nesi/project/vuw04485/slurm/tabdiff_train_%j.out
#SBATCH --error=/nesi/project/vuw04485/slurm/tabdiff_train_%j.err

# -------------------------------
# Config
# -------------------------------
DATANAME="timss_2023"
EXP_NAME="learnable_schedule"

TABDIFF_ROOT="/nesi/project/vuw04485/third_party/TabDiff"

# -------------------------------
# Environment
# -------------------------------
module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/venvs/tabdiff/bin/activate

export PYTHONPATH=${TABDIFF_ROOT}:$PYTHONPATH
cd ${TABDIFF_ROOT}

# -------------------------------
# Sanity check
# -------------------------------
python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
EOF

# -------------------------------
# Train (NO RESUME)
# -------------------------------
python main.py \
  --dataname ${DATANAME} \
  --mode train \
  --exp_name ${EXP_NAME} \
  --gpu 0 \
  --no_wandb
