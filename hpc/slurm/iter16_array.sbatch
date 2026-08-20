#!/usr/bin/env bash
#SBATCH --job-name=upgd16
#SBATCH --array=0-23%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/upgd16_%A_%a.out
#SBATCH --error=logs/upgd16_%A_%a.err

set -euo pipefail
module load cuda/12.1 2>/dev/null || true
source "${UPGD_VENV:-$HOME/.venvs/upgd}/bin/activate"
export HF_HOME="${HF_HOME:-$SCRATCH/huggingface}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs
python scripts/run_uncertainty_regularization_study.py \
  --config configs/iter16_uncertainty.yaml \
  --array-index "$SLURM_ARRAY_TASK_ID" \
  --train-only \
  --resume
