#!/usr/bin/env bash
#SBATCH --job-name=dms-ddd-cnn
#SBATCH --output=logs/slurm-ddd-%j.out
#SBATCH --error=logs/slurm-ddd-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs models reports

# Adjust these for your HPC module names if needed.
module load python/3.10 2>/dev/null || true
module load cuda 2>/dev/null || true

if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi
source .venv/bin/activate

export EPOCHS="${EPOCHS:-12}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export LR="${LR:-0.001}"

bash hpc_train.sh
