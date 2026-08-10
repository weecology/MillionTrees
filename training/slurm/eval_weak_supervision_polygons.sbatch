#!/bin/bash
#SBATCH --job-name=mt_eval_weaksup_poly
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Manuscript Table 6 (weak-supervision box pretraining vs COCO backbone) refreshed for
# the AP40 change. Re-evaluates the EXISTING polygon checkpoints from both arms -- no
# retraining -- so the comparison is unchanged apart from the dataset version (v0.22)
# and AP now scoring at IoU 0.4. Same inference path as the runs that produced the
# current numbers: training/polygons/eval.py reuses train.py's tiled predict + metrics,
# image_size 448 (the training resolution of both arms).
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
# 4 tasks = 2 init arms x 2 splits.
ARMS=(polygon_coco polygon_coco polygon_box_pretrained polygon_box_pretrained)
SPLITS=(within-distribution out-of-distribution within-distribution out-of-distribution)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/${ARM}/${SPLIT}"

# Checkpoints are named polygons-epoch=NN-val_mask_acc=*.ckpt; pick the highest accuracy
# (the selection rule the polygon leaderboard eval uses).
CKPT=$(ls -1 "$OUT_DIR/checkpoints"/polygons-epoch=*-val_mask_acc=*.ckpt \
  | awk -F'val_mask_acc=' '{print $2"\t"$0}' \
  | sort -rg \
  | head -1 \
  | cut -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no checkpoint in $OUT_DIR/checkpoints" >&2; exit 1; fi

echo "=== Weak-supervision polygon eval: arm=$ARM split=$SPLIT ==="
echo "Checkpoint: $CKPT"

uv run python training/polygons/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 4 \
  --num-workers 8 \
  --image-size 448 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz"

echo "=== done: $OUT_DIR/results_${SPLIT}.txt ==="
