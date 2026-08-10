#!/bin/bash
# Full-release 448-vs-896 run (34699513) confirmed image_size raises point recall
# (random 0.28->0.45) BUT regressed counting MAE (53->127) — opposite of the small
# set. 672 is the suspected recall/MAE balance (small: recall 0.42, MAE 70, both
# beat 448). Run pretrained TreeFormer at image_size=672 on the FULL release,
# random + zeroshot, to pick the leaderboard resolution. Inference only.
#SBATCH --job-name=mt_tf_points_full_672
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees/existing_models/treeformer
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR="outputs/full_${SPLIT}_img672"
echo "######## split=${SPLIT} image_size=672 ########"
uv run python eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --score-thresh 0.1 \
  --score-integration-radius 2 \
  --image-size 672 \
  --batch-size 8 \
  --num-workers 8 \
  --viz-dir "" \
  --output-dir "$OUT_DIR"

echo "=== Done. Results in $OUT_DIR ==="
