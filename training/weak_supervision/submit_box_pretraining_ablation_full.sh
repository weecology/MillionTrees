#!/usr/bin/env bash
# WHOLE-NETWORK same-geometry (box -> box) unsupervised pretraining ablation, v0.23.
#
# The question: does inserting a pretraining step on the 6.4M unsupervised boxes
# (Weinstein et al. 2018 unsupervised, 41,691 images) beat training on the supervised
# train split straight from torchvision COCO?
#
# Generation 4 (39129719/20/21) answered "no" — but transferred only the ResNet trunk,
# 265 of 301 tensors, resetting the FPN to COCO and both detection heads to random init.
# That truncation was inherited from the polygon (box -> Mask R-CNN) ablation, where the
# trunk genuinely is the only compatible piece; in box -> box the architectures are
# identical and every tensor transfers. This generation transfers all 301.
#
#   Stage 1: pretrain from COCO on the unsupervised boxes, export the FULL network
#   Stage 2: fine-tune that whole network on the supervised train split (depends on 1)
#
# ONLY TWO JOBS. The COCO control (39129720) is NOT re-run: stage 1 still starts from
# COCO, so that finished arm is already "the thing pretraining started from, fine-tuned."
#   COCO control, within-distribution: recall 0.637 / prec 0.633 / AP40 0.476 / MAE 14.89
#   COCO control, out-of-distribution: recall 0.668 / prec 0.777 / AP40 0.517 / MAE 15.10
#
# Design choices carried over deliberately (per user, 2026-08-11): the pretraining stage
# stays a short warmup — no held-out split, no checkpoint selection, last epoch exported.
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/weak_supervision/outputs

echo "Submitting stage 1: COCO-init pretraining with FULL network export (both splits)..."
PRETRAIN_JOB=$(sbatch --parsable training/slurm/pretrain_backbone_v023_cocoinit_full.sbatch)
echo "  Pretrain job ID: $PRETRAIN_JOB"

echo "Submitting stage 2: whole-network transfer arm (depends on stage 1)..."
FULL_JOB=$(sbatch --parsable --dependency=afterok:"$PRETRAIN_JOB" \
  training/slurm/train_boxes_box_pretrained_full_v023.sbatch)
echo "  Whole-network job ID: $FULL_JOB (waits for $PRETRAIN_JOB)"

echo ""
echo "Whole-network box pretraining ablation submitted."
echo "  Pretrain:       $PRETRAIN_JOB -> training/weak_supervision/outputs/pretrain_v023_cocoinit_full/"
echo "  Whole-network:  $FULL_JOB -> training/boxes/outputs/box_pretrained_full_v023/"
echo ""
echo "FIRST CHECK on the pretrain log: 'Verified COCO init: all 281 backbone/FPN tensors"
echo "match torchvision RetinaNet COCO_V1', then 'Full network export written to ... (301 tensors)'."
echo "FIRST CHECK on the train log: 'Loaded FULL pretrained network: 301 tensors"
echo "(trunk 265, FPN 16, heads 20)'. If it says 'Loaded 265 backbone keys' the wrong"
echo "init-mode is in play and the truncation is back."
echo ""
echo "Then compare three arms per split (Recall / Mask-aware Precision / AP40 / counting MAE):"
echo "  COCO only        training/boxes/outputs/coco_v023_cleaninit/<split>/"
echo "  trunk-only       training/boxes/outputs/box_pretrained_coco_v023_cleaninit/<split>/"
echo "  whole-network    training/boxes/outputs/box_pretrained_full_v023/<split>/"
