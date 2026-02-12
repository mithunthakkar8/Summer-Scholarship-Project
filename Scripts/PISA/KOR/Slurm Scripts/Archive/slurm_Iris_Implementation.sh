#!/bin/bash -e
#SBATCH --job-name=great_sem
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8         # Good balance for tokenization
#SBATCH --mem=32G                 # Enough for DistilGPT2 + dataset
#SBATCH --time=04:00:00      

module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/pisa_env/bin/activate

cd /nesi/project/vuw04485/pisa_sem_pipeline/python

python Iris_Data_Generation.py
