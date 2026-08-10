#!/bin/bash
#SBATCH --job-name=sam3_prompt_fig_cand
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Screening run for the good-vs-bad prompt-sensitivity figure. Row 1 is the "does well"
# source the figure will keep (Allen et al. 2025, validation split). Rows 2-4 are candidate
# "struggles" sources with middling — not catastrophic — polygon scores, so the failure is
# still legible as crowns rather than one blanket mask covering the tile:
#   SelvaMask           AP50 0.138 / recall 0.296
#   Cloutier et al. 2023 AP50 0.152 / recall 0.431
#   NEON MultiTemporal   AP50 0.064 / recall 0.552
# Pick one row from this figure, then render the final two-row version.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
cd $REPO/existing_models/sam3

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

uv run python figure_semantic_prompts_validation.py \
  --root-dir "$ROOT_DIR" --device cuda \
  --split-scheme within-distribution \
  --source-name "Allen et al. 2025" "SelvaMask" "Cloutier et al. 2023" "NEON MultiTemporal" \
  --eval-split validation test test test \
  --num-images 1 --output-tag candidates

echo "=== done -> $REPO/docs/public/sam3_prompt_sensitivity_validation_test_candidates.png ==="
