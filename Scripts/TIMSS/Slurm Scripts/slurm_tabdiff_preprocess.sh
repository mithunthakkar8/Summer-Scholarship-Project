#!/bin/bash -e
#SBATCH --job-name=tabdiff_prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nesi/project/vuw04485/slurm/tabdiff_prepare_%j.out
#SBATCH --error=/nesi/project/vuw04485/slurm/tabdiff_prepare_%j.err

DATANAME="timss_2023"
TABDIFF_ROOT="/nesi/project/vuw04485/third_party/TabDiff"

module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a

source /nesi/project/vuw04485/venvs/tabdiff/bin/activate

cd ${TABDIFF_ROOT}

# Ensure TabDiff is importable
export PYTHONPATH=${TABDIFF_ROOT}

echo "Preparing dataset: ${DATANAME}"

python process_dataset.py --dataname ${DATANAME}

echo "Dataset preparation completed"
