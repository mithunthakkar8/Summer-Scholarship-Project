#!/bin/bash -e
#SBATCH --job-name=basic_great_sem_distilgpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GA100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# ------------------------------------------------------------
# Environment detection
# ------------------------------------------------------------

echo "Running on Raapoi (VUW local cluster)"



export CLUSTER_ENV="RAAPOI"
export PYTHONPATH=/nfs/home/thakkamith/projects/TIMSS/third_party/TapTap:$PYTHONPATH

# ---- HF cache consistency (strongly recommended)
export HF_HOME=/nfs/scratch/thakkamith/hf_cache
export HF_HUB_CACHE=/nfs/scratch/thakkamith/hf_cache

module purge
module load GCCcore/12.2.0
module load Python/3.10.8

source /nfs/home/thakkamith/venvs/sem_venv/bin/activate

cd /nfs/home/thakkamith/projects/PISA/scripts

# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------
echo "Cluster environment : $CLUSTER_ENV"
echo "Python executable   : $(which python)"
echo "Python version      : $(python --version)"

python 007_python_basic_taptap_distilgpt2.py
