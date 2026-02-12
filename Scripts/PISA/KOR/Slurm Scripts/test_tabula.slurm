#!/bin/bash
#SBATCH --job-name=tabula_test
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=tabula-test-%j.out

module --force purge
module load NeSI/zen3   # correct for Milan nodes

source /nesi/project/vuw04485/pisa_env/bin/activate

export PYTHONPATH=/nesi/project/vuw04485/third_party/Tabula-main:$PYTHONPATH
export HF_HOME=/nesi/project/vuw04485/hf_cache
export HF_HUB_CACHE=/nesi/project/vuw04485/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - << 'EOF'
import os
print("Python:", os.sys.version)

from tabula import Tabula
print("Tabula import OK")
EOF
