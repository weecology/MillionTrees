#!/bin/bash
# Manuscript-final validation score for the cross-geometry TreeFormer+SAM2 pipeline:
# the v0.20 within-distribution fine-tuned TreeFormer point checkpoint (epoch 13,
# val point_recall 0.2265) + SAM2, Run-M config -- i.e. the exact leaderboard
# cross-geometry model -- evaluated on the v0.21 held-out TLS validation set
# (Allen et al. 2025 + Frey et al. 2026, 85 imgs). Same pipeline/config as
# eval_treeformer_sam2_crossgeom_v020.sbatch; only --eval-split validation and the
# output dir differ, so results are directly comparable to Detectron2/Detectree2/SAM3
# polygon validation rows in Table 5.
#SBATCH --job-name=mt_tf_sam2_val
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
export UV_NO_CACHE=1

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/validation

# v0.24 within-distribution fine-tuned TreeFormer point checkpoint (val_loss-selected,
# epoch 18) -- the exact checkpoint behind the v0.24 leaderboard cross-geometry TEST row
# (existing_models/slurm/eval_treeformer_sam2_crossgeom_v024.sbatch, job 40371591), so
# this validation score is directly comparable to it. Override with CKPT=... sbatch ...
CKPT="${CKPT:-$REPO/training/points/outputs/within-distribution_896_countfix/checkpoints/treeformer-epoch=18-val_loss=1.7955.ckpt}"
if [ ! -f "$CKPT" ]; then echo "ERROR: checkpoint not found: $CKPT" >&2; exit 1; fi

cd $REPO/existing_models/treeformer_sam2

echo "=== TreeFormer + SAM2 cross-geometry pipeline on validation (Allen+Frey) ==="
echo "checkpoint: $CKPT"

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
  --eval-split validation \
  --checkpoint "$CKPT" \
  --device cuda \
  --batch-size 16 \
  --num-workers 8 \
  --negative-prompts \
  --sam2-model "facebook/sam2.1-hiera-large" \
  --prompt-batch-size 256 \
  --score-thresh-tf 0.3 \
  --sam2-accept-score 0.1 \
  --mask-threshold 0.3 \
  --image-size 1024 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 1024

echo "=== Done. Results in $OUT_DIR ==="
