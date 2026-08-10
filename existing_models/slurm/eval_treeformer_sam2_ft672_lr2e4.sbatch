#!/bin/bash
# Fine-tuned TreeFormer (672-res point retrain, CLEAN lr2e-4 run) + SAM2, Run-M config.
# Identical to eval_treeformer_sam2_ft672.sbatch except the point checkpoint is the
# stable lr2e-4 retrain (34772934_0, epoch18 val point_recall 0.2847) instead of the
# epoch6 ckpt salvaged from the NaN-hung lr8e-4 run. Confirms whether FT at a clean,
# fully-trained 672 ckpt changes the cross-geometry result vs pretrained Run M.
#SBATCH --job-name=mt_tf_sam2_ft672_lr2e4
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
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/crossgeometry_ft672_lr2e4

# Clean 672-res point checkpoint (epoch 18, val point_recall 0.2847) from the stable
# lr2e-4 retrain (job 34772934_0); fully trained, no NaN/DDP hang.
CKPT="$REPO/training/points/outputs/random_672/checkpoints/treeformer-epoch=18-point_recall=0.2847.ckpt"

cd $REPO/existing_models/treeformer_sam2

echo "=== FT-672 (lr2e-4 clean) TreeFormer + SAM2 (Run-M config) ==="
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
  --iou-threshold 0.3 \
  --mask-threshold 0.3 \
  --image-size 1024 \
  --profile \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 1024

echo "=== Done. Results in $OUT_DIR ==="
