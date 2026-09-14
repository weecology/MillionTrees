#!/bin/bash
#SBATCH --job-name=mt_pretrain_maskrcnn_smoke
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Smoke test for the Mask R-CNN stage-1 arm. Mini dataset, 1 epoch, 2 batches --
# this exists to prove the plumbing before a 24h run burns a GPU day on it:
#
#   * the polygon config builds and assert_coco_trunk passes (366 tensors)
#   * disable_mask_branch works and no loss_mask appears
#   * use_box_task_metrics gives real box_recall (the --ckpt-monitor metric)
#   * a full train + validate + MillionTrees eval + export cycle completes
#   * the 404-tensor export lands on disk and stage 2 accepts it
#
# The final check is the one that matters most: it runs train.py's loader against
# the export, because a stage-1 run that finishes and writes an unloadable file
# is the expensive failure mode.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

OUT_DIR="training/weak_supervision/outputs/pretrain_maskrcnn_smoke"
rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"

uv run python training/boxes/pretrain_backbone_for_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme within-distribution \
  --arch maskrcnn \
  --init-mode coco \
  --trainable-backbone-layers 5 \
  --mini \
  --batch-size 4 \
  --max-epochs 1 \
  --lr 1e-3 \
  --scheduler cosine \
  --grad-clip 1.0 \
  --supervised-val-images 8 \
  --val-every-n-epochs 1 \
  --ckpt-monitor box_recall \
  --ckpt-mode max \
  --viz-images 0 \
  --num-workers 8 \
  --limit-train-batches 2 \
  --limit-val-batches 2 \
  --eval-max-batches 2 \
  --output-dir "$OUT_DIR"

echo
echo "=== stage-2 acceptance check: can train.py load what stage 1 wrote? ==="
uv run python - <<'PY'
from deepforest import main as df_main
from training.boxes.pretrain_backbone_for_polygons import POLYGON_CONFIG
from training.polygons.train import (assert_coco_trunk,
                                     apply_box_pretrained_maskrcnn)

over = {"model": {"name": None},
        "train": {"csv_file": None, "root_dir": "."},
        "validation": {"csv_file": None, "root_dir": "."},
        "maskrcnn": {"trainable_backbone_layers": 3}}
m = df_main.deepforest(config=POLYGON_CONFIG, config_args=over)
assert_coco_trunk(m)
d = apply_box_pretrained_maskrcnn(
    m,
    "training/weak_supervision/outputs/pretrain_maskrcnn_smoke/"
    "maskrcnn_transferable_within-distribution.pt")
assert d["loaded_keys"] == 404, d
assert d["mask_head_keys_at_coco"] == 28, d
print("STAGE-2 ACCEPTANCE OK:", d)
PY
echo "SMOKE PASSED"
