#!/bin/bash
#SBATCH --job-name=mt_pretrain_maskrcnn
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# STAGE 1 of the Mask R-CNN weak-supervision arm: pretrain the POLYGON
# architecture on the 6.4M unsupervised boxes, so that stage 2 inherits a whole
# tree detector instead of a bare ResNet trunk.
#
# WHY THIS EXISTS. Every polygon version of the weak-supervision ablation so far
# has been structurally truncated. Stage 1 was a RetinaNet, and RetinaNet has no
# RPN, no ROI box head and no mask head to donate, so the only architecturally
# compatible piece was backbone.body -- and in practice only 265 of that body's
# 318 tensors actually landed, leaving the FPN, the RPN and both ROI heads at
# COCO or random init. Every layer encoding "this is a tree and this is its
# extent" was discarded on transfer. That truncation is the standing explanation
# for the polygon null (notes/weak_supervision_pretraining_table.md, section 3),
# and on the BOX side it was the whole story: trunk-only transfer scored -0.016
# AP40, while the 301-tensor whole-network transfer landed exactly on the COCO
# baseline. The polygon side never got its whole-network arm because none was
# architecturally possible.
#
# This makes one possible. Stage 1 now builds the same model stage 2 builds --
# deepforest_polygon.yaml's Mask R-CNN v2, num_classes=1 -- and trains it on
# boxes with the mask branch disabled (roi_heads.mask_roi_pool = None, so
# has_mask() is false and no target ever needs a "masks" key). Weak boxes carry
# no mask supervision, and synthesising rectangular masks from them would be
# worse than useless: it would teach the mask head to emit rectangles and then
# transfer that into stage 2.
#
# WHAT TRANSFERS: 404 of 432 tensors.
#   backbone.body 318 + backbone.fpn 48 + rpn 8
#   + roi_heads.box_head 26 + roi_heads.box_predictor 4
# Only the 28-tensor mask head stays at COCO, and it must -- stage 1 never
# trains it. Compare 265 for the arm this replaces.
#
# --trainable-backbone-layers 5 (vs. the polygon config's 3). The config's
# Detectron2 FREEZE_AT=2 analog freezes stem + layer1, which is right for stage 2
# and wrong for stage 1: freezing the stem there means the weak boxes can never
# reach it. Training all five here means stage 2 then freezes those layers at
# weak-label-adapted values instead of at COCO.
#
# Recipe is otherwise byte-for-byte the converged stage-1 recipe from
# pretrain_backbone_converged.sbatch (job 39974598) -- lr 1e-3, cosine, grad-clip
# 1.0, 512 held-out SUPERVISED val images, --ckpt-monitor box_recall --ckpt-mode
# max -- because those three fixes are exactly what the stage-1 audit
# (notes/weak_supervision_stage1_audit.md) found missing. Do not drop any of
# them: without the clip lr1e-3 diverged late, and selecting on val_loss exported
# an epoch with recall 0.003.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}
# Unbuffered stdout. Without this, python's stdout is block-buffered when
# redirected to a file while stderr is not, so the .out log stays EMPTY for
# hours on a long run while .err fills -- which silently defeats the
# 'check the .out to confirm it is running' step in CLAUDE.md section 1.
export PYTHONUNBUFFERED=1

# One split per submission, NOT an array. The notes' hard-won lesson: chaining a
# whole array behind one afterok meant one task's quota death cancelled both
# splits (submit_box_pretraining_ablation_full.sh). submit_maskrcnn_pretraining.sh
# submits the two splits as independent jobs for the same reason.
SPLIT="${SPLIT:-within-distribution}"
TAG=maskrcnn-lr1e-3-cosine-clip
OUT_DIR="training/weak_supervision/outputs/pretrain_maskrcnn/$TAG/$SPLIT"
mkdir -p "$OUT_DIR"

uv run python training/boxes/pretrain_backbone_for_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --arch maskrcnn \
  --include-unsupervised \
  --init-mode coco \
  --trainable-backbone-layers 5 \
  --batch-size 32 \
  --max-epochs 20 \
  --lr 1e-3 \
  --scheduler cosine \
  --grad-clip 1.0 \
  --supervised-val-images 512 \
  --val-every-n-epochs 1 \
  --ckpt-monitor box_recall \
  --ckpt-mode max \
  --viz-images 8 \
  --num-workers 16 \
  --output-dir "$OUT_DIR" \
  --comet-name "boxes-${SPLIT}-pretrain-maskrcnn-${TAG}" \
  --comet
