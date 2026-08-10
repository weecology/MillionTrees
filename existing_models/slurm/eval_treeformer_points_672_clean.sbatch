#!/bin/bash
# Bohlman-clean pretrained TreeFormer point eval at the chosen leaderboard
# resolution (image_size=672), full TreePoints v0.18, random + zeroshot.
# The existing full_*_img672 dirs (job 34701796) predate the Bohlman 2008
# removal, so they still include the broken source. This re-runs pretrained
# @672 now that Bohlman is dropped from the dataset config, writing to
# *_img672_clean so the pre-Bohlman dirs are preserved. Feeds the manuscript
# table3 "No" point rows alongside the Bohlman-clean finetuned eval.
# Inference only.
#SBATCH --job-name=mt_tf_points_672_clean
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

OUT_DIR="outputs/full_${SPLIT}_img672_clean"
echo "######## PRETRAINED (Bohlman-clean) split=${SPLIT} image_size=672 ########"
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
