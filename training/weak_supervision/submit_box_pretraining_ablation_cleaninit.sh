#!/usr/bin/env bash
# CORRECTED same-geometry (box -> box) unsupervised pretraining ablation, v0.23.
#
# Replaces submit_box_pretraining_ablation.sh (jobs 38840963/38840964/38840965), whose
# result — "pretraining does not help the box task" — was an artifact, not a finding.
# All three stages there silently started from weecology/deepforest-tree: DeepForest's
# config default is model.name = "weecology/deepforest-tree" and deepforest.__init__
# loads it during create_model(), so declining to call load_model() left the NEON-trained
# detector in place rather than torchvision COCO. Measured consequence: the exported
# "box-pretrained" trunk sat 6.4% (median relative norm) from deepforest-tree, while
# deepforest-tree itself sits 18.3% from COCO. Both arms were tree detectors already.
#
# training/boxes/train.py now sets model.name=None for --init-mode coco/box_pretrained
# and calls assert_coco_trunk() to verify against torchvision COCO_V1 before training;
# pretrain_backbone_for_polygons.py gained --init-mode with the same guard.
#
#   Stage 1:  pretrain trunk from COCO on TreeBoxes + unsupervised  (array over splits)
#   Stage 2a: RetinaNet with that trunk, COCO FPN/heads             (depends on stage 1)
#   Stage 2b: RetinaNet with plain COCO init                        (runs in parallel)
#
# New output dirs throughout, so the contaminated 08-07 runs stay reproducible.
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/weak_supervision/outputs

echo "Submitting COCO-init v0.23 backbone pretraining (both splits)..."
PRETRAIN_JOB=$(sbatch --parsable training/slurm/pretrain_backbone_v023_cocoinit.sbatch)
echo "  Pretrain job ID: $PRETRAIN_JOB"

echo "Submitting true-COCO RetinaNet control arm (runs in parallel)..."
COCO_JOB=$(sbatch --parsable training/slurm/train_boxes_coco_v023_cleaninit.sbatch)
echo "  COCO control job ID: $COCO_JOB"

echo "Submitting box-pretrained RetinaNet treatment arm (depends on pretrain)..."
BOXINIT_JOB=$(sbatch --parsable --dependency=afterok:"$PRETRAIN_JOB" \
  training/slurm/train_boxes_box_pretrained_coco_v023_cleaninit.sbatch)
echo "  Box-init job ID: $BOXINIT_JOB (waits for $PRETRAIN_JOB)"

echo ""
echo "Corrected box pretraining ablation submitted."
echo "  Pretrain backbone:  $PRETRAIN_JOB -> training/weak_supervision/outputs/pretrain_v023_cocoinit/"
echo "  COCO control:       $COCO_JOB -> training/boxes/outputs/coco_v023_cleaninit/"
echo "  Box-pretrained:     $BOXINIT_JOB -> training/boxes/outputs/box_pretrained_coco_v023_cleaninit/"
echo ""
echo "FIRST CHECK on every log: the line 'Verified COCO init: all 281 backbone/FPN tensors"
echo "match torchvision RetinaNet COCO_V1'. If it is absent, the contamination is back."
echo "Then compare Recall / Mask-aware Precision / AP40 per split from results_<split>.txt."
