#!/bin/bash
#SBATCH --job-name=mt_eval_boxes_dome
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Evaluate trained Dome-DETR checkpoints at score_threshold=0.1 (leaderboard standard).
# Runs in the same isolated environment as training.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /blue/ewhite/b.weinstein/src/MillionTrees-dome-detr

# Activate the isolated Dome-DETR environment
source training/boxes_dome_detr/.venv/bin/activate

SPLITS=(within-distribution out-of-distribution)
for SPLIT in "${SPLITS[@]}"; do
  OUTPUT_DIR="training/boxes_dome_detr/outputs/$SPLIT"
  if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Skipping $SPLIT (no output dir found)"
    continue
  fi

  # Find the latest checkpoint (best model saved by Dome-DETR)
  CHECKPOINT=$(ls -1t "$OUTPUT_DIR"/*.pth 2>/dev/null | head -1)
  if [ -z "$CHECKPOINT" ]; then
    echo "No checkpoint found for $SPLIT"
    continue
  fi

  echo "Evaluating $SPLIT with checkpoint: $CHECKPOINT"
  python training/boxes_dome_detr/eval.py \
    --checkpoint "$CHECKPOINT" \
    --split-scheme "$SPLIT" \
    --eval-split test \
    --batch-size 8 \
    --num-workers 4 \
    --score-threshold 0.1 \
    --output-dir "$OUTPUT_DIR" \
    --viz-dir "$OUTPUT_DIR/viz"
done
