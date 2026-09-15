#!/bin/bash
#SBATCH --job-name=mt_sweep_poly_d2_ood_thresh
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --partition=hpg-default

# Offline score-threshold sweep of the Detectron2 Mask R-CNN polygon OOD prediction
# dump produced by eval_polygons_detectron2_ood_dump.sbatch (job 40694716). Re-scores
# recall / mask-aware precision / F1 across the eval-score-threshold grid from ONE
# inference dump. CPU-only: rasterizing the v0.24 polygon masks for every test image
# is memory heavy (OOM-kills a laptop-class host), so run it on a fat CPU node.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
DUMP="$REPO/outputs/threshold_sweep_preds/detectron2_polygons_out-of-distribution_test.pkl"
DATA_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}/TreePolygons_v0.24

if [ ! -f "$DUMP" ]; then
  echo "ERROR: missing prediction dump $DUMP" >&2; exit 1
fi
if [ ! -d "$DATA_DIR/masks" ]; then
  echo "ERROR: missing $DATA_DIR/masks" >&2; exit 1
fi

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"

echo "=== Detectron2 polygon OOD score-threshold sweep ==="
echo "Dump:     $DUMP"
echo "Data dir: $DATA_DIR"

uv run python -m scripts.sweep_score_threshold \
  --predictions "$DUMP" \
  --data-dir "$DATA_DIR" \
  --out-dir "$REPO/notes"

echo "=== done: notes/detectron2_polygons_out-of-distribution_test_score_threshold_sweep.csv ==="
