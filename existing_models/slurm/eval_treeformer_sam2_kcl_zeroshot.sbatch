#!/bin/bash
# KCL checkpoint: local fine-tuned TreeFormer HF checkpoint
# TreePolygons zeroshot test split
#SBATCH --job-name=mt_tf_sam2_kcl_zeroshot
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
CHECKPOINT=/home/veitchmichaelisj/code/DeepForest_jvm/kcl_hf_checkpoint
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/zeroshot_kcl

cd $REPO/existing_models/treeformer_sam2

echo "=== TreeFormer+SAM2 polygon eval (KCL checkpoint) split=zeroshot ==="
echo "Root dir: $ROOT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUT_DIR"

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme zeroshot \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --batch-size 1 \
  --num-workers 8 \
  --iou-threshold 0.3 \
  --mask-threshold 0.3 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 1024

echo "=== Done. Results in $OUT_DIR ==="
