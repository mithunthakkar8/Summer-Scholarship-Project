#!/bin/bash -e
#SBATCH --job-name=tabdiff_sample
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/nesi/project/vuw04485/slurm/tabdiff_sample_%j.out
#SBATCH --error=/nesi/project/vuw04485/slurm/tabdiff_sample_%j.err

# -------------------------------
# Config
# -------------------------------
DATANAME="timss_2023"
EXP_NAME="learnable_schedule"
NUM_SAMPLES=341
NUM_RUNS=5

TABDIFF_ROOT="/nesi/project/vuw04485/third_party/TabDiff"

# -------------------------------
# Cache redirection
# -------------------------------
export HF_HOME=/nesi/project/vuw04485/hf_cache
export TRANSFORMERS_CACHE=/nesi/project/vuw04485/hf_cache
export TORCH_HOME=/nesi/project/vuw04485/torch_cache
export XDG_CACHE_HOME=/nesi/project/vuw04485/.cache


# -------------------------------
# Environment
# -------------------------------
module --force purge
module load NeSI/zen3
module load Python/3.10.5-gimkl-2022a
source /nesi/project/vuw04485/venvs/tabdiff/bin/activate

export PYTHONPATH=${TABDIFF_ROOT}:$PYTHONPATH
cd ${TABDIFF_ROOT}
export CUDA_VISIBLE_DEVICES=""


# -------------------------------
# Sampling (THE IMPORTANT PART)
# -------------------------------
for i in $(seq 1 ${NUM_RUNS}); do
  echo "=== Sampling run ${i} ==="

  export TABDIFF_SAMPLE_TAG="run_${i}"

  python main.py \
    --dataname ${DATANAME} \
    --mode test \
    --exp_name ${EXP_NAME} \
    --num_samples_to_generate ${NUM_SAMPLES} \
    --gpu 0 \
    --no_wandb
done

