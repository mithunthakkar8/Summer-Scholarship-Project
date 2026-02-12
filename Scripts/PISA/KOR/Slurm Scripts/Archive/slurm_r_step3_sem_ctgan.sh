#!/bin/bash -e
#SBATCH --job-name=pisa_sem_ctgan
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module --force purge
module load NeSI/zen3
module load R/4.3.1-gimkl-2022a

cd /nesi/project/vuw04485/pisa_sem_pipeline/r

Rscript r_step3_sem_ctgan.R
