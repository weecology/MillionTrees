#!/usr/bin/env bash
# Same-geometry (box -> box) unsupervised pretraining ablation, v0.23.
#
# Counterpart to submit_weak_supervision.sh, which runs the polygon (box -> polygon)
# ablation for manuscript Table 6. This one asks the same question on the BOX task with
# a DeepForest RetinaNet: does an unsupervised box-pretrained ResNet trunk beat a plain
# COCO trunk?
#
#   Stage 1: pretrain backbone on TreeBoxes + unsupervised   (array over both splits)
#   Stage 2a: RetinaNet with that backbone, COCO heads       (depends on stage 1)
#   Stage 2b: RetinaNet with plain COCO init                 (runs in parallel)
#
# Both stage-2 arms use --gradient-clip-val 1.0; the v0.20 attempt of the box-pretrained
# arm (36058068_0) diverged to NaN at lr 0.01 without it.
#
# No --version anywhere: all three read the newest _versions_dict key (0.23).
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/weak_supervision/outputs

echo "Submitting v0.23 box backbone pretraining (within-distribution + out-of-distribution)..."
PRETRAIN_JOB=$(sbatch --parsable training/slurm/pretrain_backbone_v023.sbatch)
echo "  Pretrain job ID: $PRETRAIN_JOB"

echo "Submitting clean COCO RetinaNet control arm (runs in parallel)..."
COCO_JOB=$(sbatch --parsable training/slurm/train_boxes_coco_v023.sbatch)
echo "  COCO control job ID: $COCO_JOB"

echo "Submitting box-pretrained RetinaNet treatment arm (depends on pretrain)..."
BOXINIT_JOB=$(sbatch --parsable --dependency=afterok:"$PRETRAIN_JOB" \
  training/slurm/train_boxes_box_pretrained_coco_v023.sbatch)
echo "  Box-init job ID: $BOXINIT_JOB (waits for $PRETRAIN_JOB)"

echo ""
echo "Box pretraining ablation submitted."
echo "  Pretrain backbone:   $PRETRAIN_JOB -> training/weak_supervision/outputs/pretrain_v023/"
echo "  COCO control:        $COCO_JOB     -> training/boxes/outputs/coco_v023/"
echo "  Box-pretrained:      $BOXINIT_JOB  -> training/boxes/outputs/box_pretrained_coco_v023/"
echo ""
echo "When all four training tasks finish, compare Recall / Mask-aware Precision / AP40"
echo "per split from results_<split>.txt in the two outputs/ dirs."
