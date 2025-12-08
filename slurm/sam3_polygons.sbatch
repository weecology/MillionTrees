#!/bin/bash
#SBATCH --job-name=mt_sam3_polygons
#SBATCH --output=logs/slurm/sam3_polygons_%x_%A_%a.out
#SBATCH --error=logs/slurm/sam3_polygons_%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p logs/slurm

if [[ -f .hf_token ]]; then
  export HF_TOKEN="$(cat .hf_token)"
fi

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot crossgeometry)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

uv run python docs/examples/sam3_polygons.py \
  --root-dir "$ROOT_DIR" \
  --device cuda \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --hf-token "$HF_TOKEN"


