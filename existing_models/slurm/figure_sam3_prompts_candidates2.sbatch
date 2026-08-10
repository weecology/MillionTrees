#!/bin/bash
#SBATCH --job-name=sam3_prompt_fig_cand2
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Screening round 2 for the good-vs-bad prompt-sensitivity figure. Round 1 (38161295) failed
# on legibility, not scores: SelvaMask/Cloutier/NEON MultiTemporal are wall-to-wall closed
# canopy, so every overlay is a flat colour wash with no visible crown boundaries. These
# candidates are open-canopy like Allen — trees separated by bare ground — so a merged or
# missed crown is actually visible:
#   Justdiggit 2023       AP50 0.183 / recall 0.321
#   Jansen et al. 2023    AP50 0.112 / recall 0.413
#   Takeshige et al. 2025 AP50 0.087 / recall 0.556
# Pick one row, then render the final two-row figure.

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
  --source-name "Allen et al. 2025" "Justdiggit 2023" "Jansen et al. 2023" "Takeshige et al. 2025" \
  --eval-split validation test test test \
  --num-images 1 --output-tag candidates_open

echo "=== done -> $REPO/docs/public/sam3_prompt_sensitivity_validation_test_candidates_open.png ==="
