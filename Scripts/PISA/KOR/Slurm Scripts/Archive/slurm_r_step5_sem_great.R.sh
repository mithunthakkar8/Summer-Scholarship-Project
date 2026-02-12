#!/bin/bash -e
#SBATCH --job-name=pisa_sem_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

module --force purge
module load NeSI/zen3
module load R/4.3.1-gimkl-2022a

cd /nesi/project/vuw04485/pisa_sem_pipeline/r

Rscript r_step5_sem_great.R
