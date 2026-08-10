#!/bin/bash
# E3+E5 (notes/treeformer_points_diagnosis_plan.md): the score-threshold sweep
# (job 34671792) showed recall is capped ~0.27 even with no score filter, so the
# model is UNDER-DETECTING, not being over-filtered. Test the two physical levers:
#   - per-source recall breakdown (E3): is loss uniform, or driven by a few
#     sources with a bad GSD/threshold?
#   - image_size / density-grid resolution (E5): finer grid => separates dense
#     canopy => more peaks.
# Each image_size is a fresh inference pass (changes model output), but the small
# test split is 47 images so the whole grid is minutes. eval_score_threshold
# pinned low (0.1) so the score filter is not the bottleneck; full per-source
# results_str is printed/saved per size.
#SBATCH --job-name=mt_tf_points_small_imgsize
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees/existing_models/treeformer
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}

for IMG in 448 672 896 1280; do
  for RADIUS in 2 1; do
    OUT_DIR="outputs/small_random_img${IMG}_r${RADIUS}"
    echo "######## image_size=${IMG} integration_radius=${RADIUS} ########"
    uv run python eval_points.py \
      --root-dir "$ROOT_DIR" \
      --small \
      --split-scheme random \
      --score-thresh 0.1 \
      --score-integration-radius "$RADIUS" \
      --image-size "$IMG" \
      --eval-score-threshold 0.1 \
      --batch-size 8 \
      --num-workers 8 \
      --viz-dir "" \
      --output-dir "$OUT_DIR"
  done
done

echo "=== Done. Per-source results in outputs/small_random_img*/ ==="
