#!/bin/bash
#SBATCH --job-name=sam3_prompt_fig_final
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Final good-vs-bad SAM3 prompt-sensitivity figure: prompts "tree" / "tree canopy" /
# "individual tree canopy" on two contrasting sources.
#   Allen et al. 2025    (validation) — open canopy, SAM3 recovers most crowns
#   Cloutier et al. 2023 (test)       — AP50 0.152, closed canopy; "tree canopy" collapses
#                                       to 8 masks over the whole tile (recall 0.16)
# Image choice is deterministic (GT count nearest --target-gt 25), so this reproduces rows
# 1 and 3 of the screening figure (38161295): Allen idx 10565, Cloutier idx 3764.
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
  --source-name "Allen et al. 2025" "Cloutier et al. 2023" \
  --eval-split validation test \
  --num-images 1 --output-tag allen_vs_cloutier

echo "=== done -> $REPO/docs/public/sam3_prompt_sensitivity_validation_test_allen_vs_cloutier.{png,svg,csv} ==="
