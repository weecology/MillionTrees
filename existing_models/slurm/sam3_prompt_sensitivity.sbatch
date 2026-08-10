#!/bin/bash
# SAM3 semantic-prompt sensitivity on one BAMFOREST (Troles et al. 2024) test image.
# Runs prompts "individual tree canopy" / "tree" / "canopy"; emits an overlay figure
# + recall CSV to docs/public/.
#SBATCH --job-name=mt_sam3_prompt_sens
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}

cd "$REPO/existing_models/sam3"

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== SAM3 semantic-prompt sensitivity (BAMFOREST) ==="
uv run python experiment_semantic_prompts.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme within-distribution \
  --device cuda \
  --output-dir "$REPO/docs/public"

echo "=== Done -> $REPO/docs/public/sam3_prompt_sensitivity_bamforest.{png,csv} ==="
