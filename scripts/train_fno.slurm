#!/bin/bash
#SBATCH --job-name=heat-fno
#SBATCH --output=logs/fno-slurm-%j.out
#SBATCH --error=logs/fno-slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:1

set -euo pipefail

mkdir -p logs checkpoints

# Adjust these lines for the MIT Engaging environment you use.
# Examples:
# module load python/3.10
# source .venv/bin/activate
# conda activate heat-surrogate

python3 src/train_fno.py \
  --epochs 100 \
  --batch-size 64 \
  --num-workers 4 \
  --device cuda \
  --fno-width 32 \
  --fno-modes 12 \
  --fno-depth 4 \
  --run-name fno_slurm_${SLURM_JOB_ID}
