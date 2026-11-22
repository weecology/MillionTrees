#!/bin/bash
#SBATCH --job-name=mt_eval_canopygis
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
# eval_polygons.py streams metrics batch-by-batch (TreePolygonsStreamingEvalState)
# and CanopyGIS v1.8.1 peaks under 5 GB GPU VRAM; 32G RAM gives headroom for
# tile decoding and dataloader workers without over-requesting QOS memory.
#SBATCH --mem=32G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=${MT_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/canopygis/${MT_OUT_SUBDIR:-outputs}/$SPLIT

cd "$REPO/existing_models/canopygis"

# SAM3 (facebook/sam3) is gated; authenticate from an untracked token file when
# HF_TOKEN is not already set in the environment.
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$REPO/.hf_token}
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

echo "=== CanopyGIS v1.8.1 eval: split=$SPLIT ==="

uv run python eval_polygons.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --device cuda \
  --batch-size 1 \
  --score-threshold 0.10 \
  --image-size 1024 \
  --output-dir "$OUT_BASE" \
  --viz-dir "$OUT_BASE/viz_polygons" \
  --viz-n-per-source 2

echo "=== CanopyGIS v1.8.1 eval done. Results in $OUT_BASE ==="
