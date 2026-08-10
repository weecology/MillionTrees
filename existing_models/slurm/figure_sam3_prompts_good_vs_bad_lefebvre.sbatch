#!/bin/bash
#SBATCH --job-name=sam3_prompt_fig_gvb2
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Same figure as figure_sam3_prompts_good_vs_bad.sbatch, but the "SAM3 does well" source is
# Lefebvre et al. 2024 (AP50 0.414 / recall 0.718) instead of Safonova et al. 2021: Safonova
# scores highest (AP50 0.898) only because every test image holds exactly 2 widely spaced
# trees, which reads as a trivial panel. Lefebvre has realistic crown density (median 9,
# selected image 25 GT). Failure source unchanged: NEON combined crowns (AP50 0.000).
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
  --split-scheme within-distribution --eval-split test \
  --source-name "Lefebvre et al. 2024" "NEON combined crowns" \
  --num-images 1 --output-tag good_vs_bad_lefebvre

echo "=== done -> $REPO/docs/public/sam3_prompt_sensitivity_test_good_vs_bad_lefebvre.{png,csv} ==="
