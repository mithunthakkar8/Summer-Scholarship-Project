#!/bin/bash -e
#SBATCH --job-name=pisa_sem_real
#SBATCH --cpus-per-task=4
#SBATCH --partition=milan
#SBATCH --mem=16G
#SBATCH --time=02:00:00
export R_LIBS_USER=$HOME/R/library

module --force purge
module load NeSI/zen3
module load R/4.3.1-gimkl-2022a

cd /nesi/project/vuw04485/pisa_sem_pipeline/r

Rscript 001_r_sem_real_using_seminr.R
