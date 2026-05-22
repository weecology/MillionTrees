#!/bin/bash
#SBATCH --job-name=mt_eval_polygons
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# Best epoch=04 checkpoints from the current v0.13 run (cancelled after 4 epochs)
declare -A BOX_CKPTS
BOX_CKPTS[random]="training/weak_supervision/outputs/polygon_box_pretrained/random/checkpoints/polygons-epoch=04-val_loss=-7045186.0000.ckpt"
BOX_CKPTS[zeroshot]="training/weak_supervision/outputs/polygon_box_pretrained/zeroshot/checkpoints/polygons-epoch=04-val_loss=-4772353.5000.ckpt"

# Best epoch=39 COCO checkpoints from the completed v0.13 run
declare -A COCO_CKPTS
COCO_CKPTS[random]="training/weak_supervision/outputs/polygon_coco/random/checkpoints/polygons-epoch=39-val_loss=-465924.2500.ckpt"
COCO_CKPTS[zeroshot]="training/weak_supervision/outputs/polygon_coco/zeroshot/checkpoints/polygons-epoch=39-val_loss=-459748.0312.ckpt"

OUT_BOX="training/weak_supervision/outputs/eval_current/polygon_box_pretrained/$SPLIT"
OUT_COCO="training/weak_supervision/outputs/eval_current/polygon_coco/$SPLIT"
mkdir -p "$OUT_BOX" "$OUT_COCO"

echo "=== Eval box_pretrained ($SPLIT) ==="
uv run python training/polygons/eval_checkpoint.py \
  --checkpoint "${BOX_CKPTS[$SPLIT]}" \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --num-workers 4 \
  --image-size 448 \
  --init-mode box_pretrained \
  --data-scope subset \
  --output-dir "$OUT_BOX"

echo "=== Eval coco ($SPLIT) ==="
uv run python training/polygons/eval_checkpoint.py \
  --checkpoint "${COCO_CKPTS[$SPLIT]}" \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --num-workers 4 \
  --image-size 448 \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_COCO"
