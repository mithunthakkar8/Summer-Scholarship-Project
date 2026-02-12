#!/bin/bash -e
#SBATCH --job-name=tabsyn_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GA100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/11.7

source /nfs/home/thakkamith/venvs/tabsyn_venv/bin/activate

export HF_HOME=/nfs/scratch/thakkamith/hf_cache
export TORCH_HOME=/nfs/scratch/thakkamith/hf_cache

cd /nfs/home/thakkamith/projects/TIMSS/third_party/TabSyn

python main.py --dataname PISA_SGP --method vae --mode train

