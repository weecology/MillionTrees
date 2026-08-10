#!/bin/bash
# E0+E1 (docs/treeformer_points_diagnosis_plan.md): pretrained TreeFormer on
# SmallTreePoints v0.18, random split. Single inference pass, then sweep the
# eval_score_threshold grid. The t=0.40 row reproduces the leaderboard recall
# (~0.28 = E0); the full curve tests whether recall recovers below 0.4 (E1).
# Emission threshold (--score-thresh) is driven to 0.05 so the evaluator, not
# the model, controls the low end of the sweep.
#SBATCH --job-name=mt_tf_points_small_sweep
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees/existing_models/treeformer
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR="outputs/small_random_sweep"

echo "=== E0+E1: TreeFormer pretrained, SmallTreePoints random, threshold sweep ==="
uv run python eval_points.py \
  --root-dir "$ROOT_DIR" \
  --small \
  --split-scheme random \
  --score-thresh 0.05 \
  --score-integration-radius 2 \
  --image-size 448 \
  --batch-size 16 \
  --num-workers 8 \
  --sweep \
  --output-dir "$OUT_DIR"

echo "=== Done. Sweep CSV in $OUT_DIR/threshold_sweep.csv ==="
