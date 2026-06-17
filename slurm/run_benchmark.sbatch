#!/bin/bash
# Master benchmark job: trains all models, runs all evaluations, generates comparison tables.
#
# Usage:
#   sbatch slurm/run_benchmark.sbatch
#
# Each stage (training, existing-model eval) is submitted as a dependent job so the
# table generation only runs after all evaluations complete.
#
# Set MT_ROOT to point at the MillionTrees dataset directory before submitting:
#   MT_ROOT=/path/to/data sbatch slurm/run_benchmark.sbatch

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
cd "$REPO"

echo "=== Submitting training jobs ==="
BOXES_JID=$(sbatch --parsable training/slurm/train_boxes.sbatch)
POINTS_JID=$(sbatch --parsable training/slurm/train_points.sbatch)
POLYS_JID=$(sbatch --parsable training/slurm/train_polygons.sbatch)
echo "  train_boxes:    $BOXES_JID"
echo "  train_points:   $POINTS_JID"
echo "  train_polygons: $POLYS_JID"

echo "=== Submitting existing-model eval jobs ==="
DF_JID=$(sbatch --parsable existing_models/slurm/eval_deepforest.sbatch)
SAM3_JID=$(sbatch --parsable existing_models/slurm/eval_sam3.sbatch)
echo "  eval_deepforest: $DF_JID"
echo "  eval_sam3:       $SAM3_JID"

TRAIN_DEPS="${BOXES_JID}:${POINTS_JID}:${POLYS_JID}"
ALL_DEPS="${TRAIN_DEPS}:${DF_JID}:${SAM3_JID}"

echo "=== Submitting fine-tuned panel figures (depends on training) ==="
VIZ_JID=$(sbatch --parsable \
  --dependency=afterany:"$TRAIN_DEPS" \
  slurm/visualize_finetuned.sbatch)
echo "  visualize_finetuned: $VIZ_JID"

echo "=== Submitting table generation (depends on training + eval) ==="
TABLE_JID=$(sbatch --parsable \
  --job-name=mt_benchmark_table \
  --output=/home/b.weinstein/logs/mt_benchmark_table_%j.out \
  --error=/home/b.weinstein/logs/mt_benchmark_table_%j.err \
  --time=00:30:00 \
  --cpus-per-task=1 \
  --mem=8G \
  --partition=hpg-turin \
  --dependency=afterany:"$ALL_DEPS" \
  --wrap="cd $REPO && uv run python scripts/make_benchmark_table.py --splits within-distribution out-of-distribution")
echo "  make_benchmark_table: $TABLE_JID"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Metrics table -> docs/leaderboard.md (Benchmark Results section)"
echo "Panel figures -> docs/leaderboard_predictions_*.{png,svg}"
