#!/bin/bash
#SBATCH --job-name=hf_download
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=hf-download.out

module --force purge
module load NeSI/zen3
source /nesi/project/vuw04485/pisa_env/bin/activate

unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

export HF_HOME=/nesi/project/vuw04485/hf_cache

python - << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained("distilgpt2")
AutoModelForCausalLM.from_pretrained("distilgpt2")
print("distilgpt2 cached")
EOF
