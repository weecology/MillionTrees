#!/bin/bash
#SBATCH --job-name=mt_render_hayachine
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
CKPT=training/polygons/outputs/zeroshot/checkpoints/polygons-epoch=19-val_mask_acc=0.3306.ckpt
FILE=Hayachine_HY-EC1_Ortho_59_Takeshige_et_al_2025.png

echo "=== Render trained MaskRCNN overlay for $FILE (zeroshot ckpt) ==="

# 448 = how the trained model is actually evaluated / how training/polygons viz looks.
uv run python training/polygons/render_single_viz.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme zeroshot \
  --image-size 448 \
  --score-threshold 0.1 \
  --filename "$FILE" \
  --viz-dir training/polygons/outputs/zeroshot/viz_single_448

# 1024 = matches the CanopyRS viz resolution for a side-by-side comparison.
uv run python training/polygons/render_single_viz.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme zeroshot \
  --image-size 1024 \
  --score-threshold 0.1 \
  --filename "$FILE" \
  --viz-dir training/polygons/outputs/zeroshot/viz_single_1024

echo "=== Done ==="
