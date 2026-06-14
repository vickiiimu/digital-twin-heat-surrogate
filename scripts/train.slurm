#!/bin/bash
#SBATCH --job-name=heat-cnn
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs checkpoints

# Adjust these lines for the MIT Engaging environment you use.
# Examples:
# module load python/3.10
# source .venv/bin/activate
# conda activate heat-surrogate

python3 src/train.py \
  --model cnn \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --run-name cnn_slurm_${SLURM_JOB_ID}
