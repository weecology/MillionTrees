#!/bin/bash
# Fine-tuned TreeFormer (672-res point retrain) + SAM2, Run-M config.
# Mirrors eval_treeformer_sam2_runM.sbatch (B200, image-size 1024, no tiling,
# negative prompts, hiera-large) but swaps the pretrained HF point model for the
# fine-tuned Lightning checkpoint. This is the "FT checkpoint at the codified
# defaults" run flagged as pending in docs/treeformer_sam2_experiments.md.
#SBATCH --job-name=mt_tf_sam2_ft672
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
export UV_NO_CACHE=1

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/crossgeometry_ft672_acc0.1

# Best 672-res point checkpoint (epoch 6, val point_recall 0.2885) from the
# random lr8e-4 run (Comet 37a475a9); saved before that run NaN-hung.
CKPT="$REPO/training/points/outputs/random_672_lr8e-4_hung/checkpoints/treeformer-epoch=06-point_recall=0.2885.ckpt"

cd $REPO/existing_models/treeformer_sam2

echo "=== FT-672 TreeFormer + SAM2 (Run-M config) ==="
echo "checkpoint: $CKPT"

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
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
  --profile \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 1024

echo "=== Done. Results in $OUT_DIR ==="
