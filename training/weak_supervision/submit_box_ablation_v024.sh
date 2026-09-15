#!/usr/bin/env bash
# v0.24 re-run of the box->box weak-supervision ablation (notes sections 1 & 2 of
# notes/weak_supervision_pretraining_table.md). The whole interpretation rides on the
# out-of-distribution row and v0.24 repairs the OOD split, so the v0.23 converged
# numbers are not citable.
#
# Submits, per split (within-distribution + out-of-distribution):
#   - converged stage-1 pretrain (coco init, weak boxes, grad-clip, box_recall ckpt)
#   - arm 0 coco control + arm 2 co-training      (no dependency)
#   - arm 1 box_pretrained_full                   (afterok on that split's stage 1)
# Plus the ImageNet cold-start rung (both splits, one array job, no dependency).
#
# Stage 1 is submitted per split -- never one array behind a single afterok -- so one
# split's failure cannot cancel the other (lesson from the 2026-08-11 quota wipeout).
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

echo "=== quota check ==="
blue_quota || true
echo ""

IMAGENET=$(sbatch --parsable training/slurm/train_boxes_imagenet_v024.sbatch)
echo "imagenet cold-start rung (both splits) : $IMAGENET"

declare -A PRE ARM02 ARM1
for SPLIT in within-distribution out-of-distribution; do
  PRE[$SPLIT]=$(sbatch --parsable --export=ALL,SPLIT="$SPLIT" \
    training/slurm/pretrain_backbone_converged_v024.sbatch)
  echo "stage-1 pretrain $SPLIT           : ${PRE[$SPLIT]}"

  ARM02[$SPLIT]=$(sbatch --parsable --array=0,2 --export=ALL,SPLIT="$SPLIT" \
    training/slurm/train_boxes_converged_pretrain_vs_cotrain_v024.sbatch)
  echo "stage-2 arms 0,2 $SPLIT           : ${ARM02[$SPLIT]}"

  ARM1[$SPLIT]=$(sbatch --parsable --array=1 \
    --dependency=afterok:"${PRE[$SPLIT]}" --export=ALL,SPLIT="$SPLIT" \
    training/slurm/train_boxes_converged_pretrain_vs_cotrain_v024.sbatch)
  echo "stage-2 arm 1 $SPLIT (afterok ${PRE[$SPLIT]}) : ${ARM1[$SPLIT]}"
done

cat <<SUMMARY

Submitted. Write one ledger entry per job ID:
  $IMAGENET                  imagenet cold-start rung, both splits
  ${PRE[within-distribution]}     stage-1 pretrain WD    -> blocks ${ARM1[within-distribution]}
  ${ARM02[within-distribution]}   stage-2 arms 0,2 WD
  ${ARM1[within-distribution]}    stage-2 arm 1 WD       (afterok:${PRE[within-distribution]})
  ${PRE[out-of-distribution]}     stage-1 pretrain OOD   -> blocks ${ARM1[out-of-distribution]}
  ${ARM02[out-of-distribution]}   stage-2 arms 0,2 OOD
  ${ARM1[out-of-distribution]}    stage-2 arm 1 OOD      (afterok:${PRE[out-of-distribution]})

FIRST CHECK on each log:
  pretrain : "Verified COCO init: all 281 backbone/FPN tensors match torchvision
             RetinaNet COCO_V1", peak box_recall > 0.24, "301 tensors" export.
  stage-2  : arm 0/2 "Verified COCO init: all 281 ..."; arm 1 "Loaded FULL pretrained
             network: 301 tensors"; arm 2 train-image count includes the weak tiles
             and an exclude_sources line naming only '*Young*'.
  imagenet : "assert_imagenet_trunk" passes.
Then rebuild notes/weak_supervision_pretraining_table.md sections 1-2 for v0.24.
SUMMARY
