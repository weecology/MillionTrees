#!/bin/bash
#SBATCH --job-name=mt_viz_finetuned
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
cd "$REPO"

echo "=== Fine-tuned panel figures (within-distribution + out-of-distribution) ==="

# TreeFormer point checkpoints need the treeformer DeepForest group.
uv run --group treeformer python scripts/create_finetuned_visualizations.py \
  --root-dir "$ROOT_DIR" \
  --output-dir "$REPO/docs" \
  --panel-dir "$REPO/docs/figures/finetuned_panels" \
  --device cuda

echo "=== Done: docs/leaderboard_predictions_*.{png,svg} and docs/figures/finetuned_panels/ ==="
