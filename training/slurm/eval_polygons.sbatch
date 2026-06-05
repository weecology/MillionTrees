#!/bin/bash
#SBATCH --job-name=mt_eval_polygons_ft
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

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

CKPT_DIR="training/polygons/outputs/${SPLIT}/checkpoints"
# Most-negative val_loss among converged checkpoints (epoch >= 1 guards against
# runaway first-epoch outliers in the -500M range that skew the sort).
CKPT=$(ls -1 "$CKPT_DIR"/polygons-epoch=*.ckpt \
  | awk -F'epoch=' '{split($2,a,"-"); if (a[1]+0 > 0) print}' \
  | awk -F'val_loss=' '{print $2"\t"$0}' \
  | sort -g \
  | head -1 \
  | cut -f2-)

echo "=== Eval fine-tuned polygons: split=$SPLIT ==="
echo "Checkpoint: $CKPT"

VIZ_DIR="outputs/viz/trained_polygons/${SPLIT}"
mkdir -p "$VIZ_DIR"

uv run python training/polygons/eval_checkpoint.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 4 \
  --num-workers 4 \
  --image-size 448 \
  --output-dir "training/polygons/outputs/${SPLIT}" \
  --viz-dir "$VIZ_DIR"
