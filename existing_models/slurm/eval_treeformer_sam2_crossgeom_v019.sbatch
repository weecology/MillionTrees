#!/bin/bash
# v0.19 leaderboard cross-geometry row: fine-tuned TreeFormer (v0.19 within-distribution
# point checkpoint, epoch 18, val point_recall 0.2259) + SAM2, Run-M config.
# Mirrors eval_treeformer_sam2_ft672.sbatch (B200, image-size 1024, negative prompts,
# hiera-large, sam2-accept-score 0.1) but swaps in the v0.19 within-distribution
# Lightning checkpoint per request. NOTE: that checkpoint was trained at image-size 448
# (train_points default); eval here runs the Run-M 1024 pipeline for comparability with
# the prior FT cross-geom run — switch --image-size to 448 if training-matched res is wanted.
#SBATCH --job-name=mt_tf_sam2_crossgeom_v019
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
export UV_NO_CACHE=1

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/crossgeometry

# v0.19 within-distribution fine-tuned TreeFormer point checkpoint.
CKPT="$REPO/training/points/outputs/within-distribution/checkpoints/treeformer-epoch=18-point_recall=0.2259.ckpt"

cd $REPO/existing_models/treeformer_sam2

echo "=== v0.19 within-FT TreeFormer + SAM2 cross-geometry (Run-M config) ==="
echo "checkpoint: $CKPT"

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
  --checkpoint "$CKPT" \
  --device cuda \
  --batch-size 16 \
  --num-workers 8 \
  --negative-prompts \
  --sam2-model "facebook/sam2.1-hiera-large" \
  --prompt-batch-size 256 \
  --score-thresh-tf 0.3 \
  --sam2-accept-score 0.1 \
  --mask-threshold 0.3 \
  --image-size 1024 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 1024

echo "=== Done. Results in $OUT_DIR ==="
