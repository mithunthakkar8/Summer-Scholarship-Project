#!/bin/bash -e
#SBATCH --job-name=basic_great_sem_gpt2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GA100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00



export CLUSTER_ENV="RAAPOI"

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

# ------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------

python 004_python_basic_great_pipeline_gpt2.py

