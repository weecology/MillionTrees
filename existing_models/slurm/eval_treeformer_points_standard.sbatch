#!/bin/bash
# Pretrained TreeFormer baseline, TreePoints, standardized post-proc
# (score_thresh=0.1, score_integration_radius=2). Re-run so the baseline is
# scored under the updated metric (new GSD for OFO/OSBS/Bohlman; Beery excluded
# from eval as a training-only source). Array over random + zeroshot.
#SBATCH --job-name=mt_tf_eval_points_std
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees/existing_models/treeformer
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="outputs/${SPLIT}_pretrained_std"

echo "=== TreeFormer pretrained eval (std st=0.1 r=2) split=$SPLIT ==="

uv run python eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --score-thresh 0.1 \
  --score-integration-radius 2 \
  --batch-size 16 \
  --num-workers 8 \
  --output-dir "$OUT_DIR"

echo "=== Done. Results in $OUT_DIR ==="
