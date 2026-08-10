#!/usr/bin/env bash
# Submit the two-stage weak supervision experiment:
#   1. Pretrain backbone on TreeBoxes (with unsupervised data)
#   2. Train polygons with box-pretrained backbone (after step 1) -- annotationsafecrop
#   3. Train polygons with COCO-pretrained weights (baseline) -- annotationsafecrop
#
# Both polygon arms use --train-aug annotationsafecrop so the COCO vs box-pretrained
# comparison in Table 6 is on the same augmentation recipe. The backbone pretrain
# itself is NOT affected by annotationsafecrop (it uses a raw DataLoader that bypasses
# DeepForest's dataset/augmentation stack).
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/weak_supervision/outputs

echo "Submitting backbone pretraining (within-distribution + out-of-distribution splits)..."
PRETRAIN_JOB=$(sbatch --parsable training/slurm/pretrain_backbone.sbatch)
echo "  Pretrain job ID: $PRETRAIN_JOB"

echo "Submitting polygon COCO-baseline training with annotationsafecrop (runs in parallel)..."
COCO_JOB=$(sbatch --parsable training/slurm/train_polygons_deepforest_annotationsafecrop.sbatch)
echo "  COCO baseline job ID: $COCO_JOB"

echo "Submitting polygon box-pretrained training with annotationsafecrop (depends on pretrain)..."
BOXINIT_JOB=$(sbatch --parsable --dependency=afterok:$PRETRAIN_JOB training/slurm/train_polygons_box_pretrained_annotationsafecrop.sbatch)
echo "  Box-init job ID: $BOXINIT_JOB (waits for $PRETRAIN_JOB)"

echo ""
echo "All weak supervision jobs submitted."
echo "  Pretrain backbone:      $PRETRAIN_JOB"
echo "  Polygon COCO baseline:  $COCO_JOB"
echo "  Polygon box-pretrained: $BOXINIT_JOB"
