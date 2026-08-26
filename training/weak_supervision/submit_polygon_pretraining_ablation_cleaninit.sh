#!/usr/bin/env bash
# CORRECTED polygon (Table 6) unsupervised pretraining ablation, v0.23 / AP40.
#
# Replaces the coco/box_pretrained arms of submit_weak_supervision.sh (jobs
# 38750867/38750868), whose headline "+0.083 AP40 within-distribution from unsupervised
# box pretraining" is not a valid measurement of what it claims.
#
# deepforest_polygon.yaml sets no model.name, so the base config default
# 'weecology/deepforest-tree' applied and create_model() routed into
# MaskRCNN.from_pretrained(), which builds a cold-start shell and then loads a RetinaNet
# checkpoint whose keys never match a Mask R-CNN. Nothing reached the backbone. Measured
# on the old config: 0/318 backbone.body tensors equal torchvision COCO_V1, all 53
# BatchNorm running_mean still exactly 0, and two fresh builds disagree on all 53 conv
# weights -- the "COCO baseline" ResNet-50 was RANDOMLY INITIALIZED.
#
# So Table 6 compared a pretrained backbone against an untrained one. That is a
# different (and much easier) claim than "unsupervised tree boxes beat generic COCO
# features", and it inflates the gain rather than hiding it -- the opposite direction
# from the box-side contamination bug.
#
# training/polygons/train.py now forces model.name=None in build_config() and calls
# assert_coco_trunk() (366 backbone+FPN tensors vs torchvision maskrcnn_resnet50_fpn_v2
# COCO_V1) before training, and before the backbone swap on the treatment arm.
#
#   Arm A: Mask R-CNN, plain COCO init                        (control)
#   Arm B: Mask R-CNN, backbone.body <- unsupervised box trunk (treatment)
#
# Stage 1 (backbone pretraining) is NOT re-run: job 39129719 already produced verified
# COCO-init v0.23 trunks for both splits at
# training/weak_supervision/outputs/pretrain_v023_cocoinit/. Both arms run in parallel.
#
# New output dirs throughout, so the 08-05 runs stay reproducible.
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/weak_supervision/outputs

for SPLIT in within-distribution out-of-distribution; do
  BACKBONE="training/weak_supervision/outputs/pretrain_v023_cocoinit/$SPLIT/box_backbone_${SPLIT}.pt"
  if [ ! -f "$BACKBONE" ]; then
    echo "ERROR: missing COCO-init backbone $BACKBONE (expected from job 39129719)" >&2
    exit 1
  fi
done

echo "Submitting COCO-init Mask R-CNN control arm (both splits)..."
COCO_JOB=$(sbatch --parsable training/slurm/train_polygons_coco_v023_cleaninit.sbatch)
echo "  COCO control job ID: $COCO_JOB"

echo "Submitting box-pretrained Mask R-CNN treatment arm (both splits)..."
BOXINIT_JOB=$(sbatch --parsable training/slurm/train_polygons_box_pretrained_v023_cleaninit.sbatch)
echo "  Box-init job ID: $BOXINIT_JOB"

echo ""
echo "Corrected polygon pretraining ablation submitted."
echo "  COCO control:   $COCO_JOB -> training/weak_supervision/outputs/polygon_coco_v023_cleaninit/"
echo "  Box-pretrained: $BOXINIT_JOB -> training/weak_supervision/outputs/polygon_box_pretrained_v023_cleaninit/"
echo ""
echo "FIRST CHECK on every log: 'Verified COCO init: all 366 backbone/FPN tensors match"
echo "torchvision Mask R-CNN v2 COCO_V1'. The treatment arm must ALSO show"
echo "'Loaded 265 backbone keys; 98 keys left at COCO init.' If either line is absent,"
echo "the init is wrong again -- kill the run rather than reporting it."
echo "Then write the four-row Recall / Mask-aware Precision / AP40 comparison per split"
echo "into notes/weak_supervision_pretraining_table.md and manuscript Table 6."
