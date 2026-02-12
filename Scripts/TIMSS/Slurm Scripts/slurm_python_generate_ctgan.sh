#!/bin/bash -e
#SBATCH --job-name=ctgan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GA100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8        
#SBATCH --mem=32G                 
  

module purge
module load GCCcore/12.2.0
module load Python/3.10.8

source /nfs/home/thakkamith/venvs/sem_venv/bin/activate

cd /nfs/home/thakkamith/projects/TIMSS/scripts

python 003_python_generate_ctgan.py
