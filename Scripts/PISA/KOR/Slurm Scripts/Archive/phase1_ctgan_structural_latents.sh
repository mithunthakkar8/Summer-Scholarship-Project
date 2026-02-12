#!/bin/bash -l
#SBATCH --job-name=sem_great_phase1_exo_ctgan
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=/nesi/project/vuw04485/pisa_sem_pipeline/logs/sem_phase2_great_%j.out
#SBATCH --error=/nesi/project/vuw04485/pisa_sem_pipeline/logs/sem_phase2_great_%j.err

module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/pisa_env/bin/activate

cd /nesi/project/vuw04485/pisa_sem_pipeline/python

python python_sem_great_phase1_exo_ctgan.py
