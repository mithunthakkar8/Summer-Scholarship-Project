#!/bin/bash -e
#SBATCH --job-name=tabsyn_vae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GA100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module purge
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/11.7

source /nfs/home/thakkamith/venvs/tabsyn_venv/bin/activate

export HF_HOME=/nfs/scratch/thakkamith/hf_cache
export TORCH_HOME=/nfs/scratch/thakkamith/hf_cache

cd /nfs/home/thakkamith/projects/TIMSS/third_party/TabSyn

for i in 1 2 3 4 5
do
  python main.py \
    --dataname PISA_SGP \
    --method tabsyn \
    --mode sample \
    --num-samples 856 \
    --seed $i \
    --save_path /nfs/scratch/thakkamith/PISA/outputs/tabsyn/rep_${i}.csv
done

