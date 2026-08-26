#!/bin/bash
# Submit the NATIVE DETECTRON2 polygon weak-supervision ablation for one split.
#
#   bash training/weak_supervision/submit_d2_polygon_ablation.sh within-distribution
#   bash training/weak_supervision/submit_d2_polygon_ablation.sh out-of-distribution
#
# Chains, per split:
#   pretrain_detectron2.sbatch        stage 1 on the weak Weinstein boxes  (array task)
#   train_polygons_d2_ablation.sbatch arms 0 (control) + 2 (co-training)   -- no dependency
#   train_polygons_d2_ablation.sbatch arm 1 (sequential)  afterok: stage 1
#
# Arms 0 and 2 need no stage-1 network, so they go straight into the queue rather than
# waiting behind it. Arm 1 is held by afterok because its weights do not exist yet; the
# sbatch also guards on the file itself, so a stage-1 failure fails loudly instead of
# silently training a second control.
#
# Remember the ledger: every job ID below needs its own Why/Next entry in
# /home/b.weinstein/logs/job_ledger.md.
set -euo pipefail

SPLIT="${1:-within-distribution}"
case "$SPLIT" in
  within-distribution|out-of-distribution) ;;
  *) echo "usage: $0 {within-distribution|out-of-distribution}" >&2; exit 2 ;;
esac
REPO=/blue/ewhite/b.weinstein/src/MillionTrees
SPLIT_INDEX=$([ "$SPLIT" = "within-distribution" ] && echo 0 || echo 1)

PRETRAIN_ID=$(sbatch --parsable --array="$SPLIT_INDEX" \
  "$REPO/training/slurm/pretrain_detectron2.sbatch")
echo "stage 1 (weak-box pretraining, $SPLIT): $PRETRAIN_ID"

FREE_ID=$(SPLIT="$SPLIT" sbatch --parsable --array=0,2 \
  "$REPO/training/slurm/train_polygons_d2_ablation.sbatch")
echo "arms 0 (control) + 2 (co-training), $SPLIT: $FREE_ID"

SEQ_ID=$(SPLIT="$SPLIT" sbatch --parsable --array=1 \
  --dependency="afterok:${PRETRAIN_ID%%_*}" \
  "$REPO/training/slurm/train_polygons_d2_ablation.sbatch")
echo "arm 1 (sequential), $SPLIT: $SEQ_ID  [afterok:${PRETRAIN_ID%%_*}]"
