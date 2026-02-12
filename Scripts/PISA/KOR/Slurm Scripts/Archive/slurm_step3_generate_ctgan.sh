#!/bin/bash -e
#SBATCH --job-name=pisa_ctgan
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:L4:1

module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/pisa_env/bin/activate

cd /nesi/project/vuw04485/pisa_sem_pipeline/python

python python_step3_generate_ctgan.py
