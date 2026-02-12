#!/bin/bash -e
#SBATCH --job-name=great_sem
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8         
#SBATCH --mem=32G                 
#SBATCH --time=04:00:00           

module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/pisa_env/bin/activate

cd /nesi/project/vuw04485/pisa_sem_pipeline/python
python 004_python_multiphase_great_pipeline.py

