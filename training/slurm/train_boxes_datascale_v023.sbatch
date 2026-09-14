#!/bin/bash
#SBATCH --job-name=mt_boxes_datascale_v023
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-8
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Data-scaling arm of the box -> box weak-supervision ablation, v0.23 / AP40.
#
# Motivation. At FULL supervision (13,538 train images / 796k boxes) the unsupervised
# boxes buy nothing, and that holds even for the strongest possible transfer:
#   COCO control            AP40 0.476 WD / 0.517 OOD   (39129720)
#   trunk-only  (265/301)   AP40 0.460 WD / 0.502 OOD   (39129721)
#   WHOLE net   (301/301)   AP40 0.476 WD / 0.501 OOD   (39614224 / 39614226)
# Transferring an entire trained tree detector lands exactly on the COCO baseline, which
# says the supervised set is large enough to wash the initialization out completely
# rather than that the pretrained features are worthless. This array tests that
# explanation directly by shrinking the SUPERVISED training set to one source (and to a
# tenth of one source) while holding the evaluation set fixed at the full test split.
#
# --train-sources filters the TRAIN split only (TreeBoxes.train_sources); validation and
# test are untouched, so every cell here is scored on the same 3,023 test images as the
# full-data baselines above and the numbers are directly comparable.
#
# Expectation: if the mechanism is data volume, the pretraining gain should grow as the
# supervised set shrinks, and box_pretrained_full at ~206 images should be a large,
# obvious win (it starts as a working tree detector). If even THAT is flat, the null is
# not about dataset size and the pretraining/transfer pipeline is what needs auditing.
#
# Tiers pick their epoch budget so each cell gets a comparable number of optimizer steps
# and enough early-stopping headroom (patience is in epochs, and a 206-image epoch is
# 7 steps). --val-frac 0.25 keeps the per-epoch monitoring metric cheap; the final scored
# eval always uses the full test split.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLIT=within-distribution

# tier index = task / 3, init index = task % 3
TIER_IDX=$((SLURM_ARRAY_TASK_ID / 3))
INIT_IDX=$((SLURM_ARRAY_TASK_ID % 3))

# Tier definitions: name | --train-sources pattern | train-frac | max-epochs | patience
TIER_NAMES=(weecology10 selvabox weecology)
TIER_SOURCES=('Weecology*' 'SelvaBox*' 'Weecology*')
TIER_FRACS=(0.1 1.0 1.0)
TIER_EPOCHS=(400 300 200)
TIER_PATIENCE=(40 30 20)

TIER=${TIER_NAMES[$TIER_IDX]}
SOURCES=${TIER_SOURCES[$TIER_IDX]}
FRAC=${TIER_FRACS[$TIER_IDX]}
EPOCHS=${TIER_EPOCHS[$TIER_IDX]}
PATIENCE=${TIER_PATIENCE[$TIER_IDX]}

INITS=(coco box_pretrained box_pretrained_full)
INIT=${INITS[$INIT_IDX]}

TRUNK="training/weak_supervision/outputs/pretrain_v023_cocoinit/$SPLIT/box_backbone_${SPLIT}.pt"
NETWORK="training/weak_supervision/outputs/pretrain_v023_cocoinit_full/$SPLIT/box_network_${SPLIT}.pt"

CKPT_ARGS=()
case "$INIT" in
  coco)
    ;;
  box_pretrained)
    [ -f "$TRUNK" ] || { echo "ERROR: missing trunk export $TRUNK" >&2; exit 1; }
    CKPT_ARGS=(--box-backbone-checkpoint "$TRUNK")
    ;;
  box_pretrained_full)
    [ -f "$NETWORK" ] || { echo "ERROR: missing full-network export $NETWORK" >&2; exit 1; }
    CKPT_ARGS=(--box-backbone-checkpoint "$NETWORK")
    ;;
esac

OUT_DIR="training/boxes/outputs/datascale_v023/$TIER/$INIT"
mkdir -p "$OUT_DIR"

echo "=== datascale cell: tier=$TIER sources='$SOURCES' frac=$FRAC init=$INIT epochs=$EPOCHS patience=$PATIENCE"

srun uv run python training/boxes/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --train-sources "$SOURCES" \
  --train-frac "$FRAC" \
  --val-frac 0.25 \
  --seed 0 \
  --batch-size 32 \
  --max-epochs "$EPOCHS" \
  --lr 0.01 \
  --num-workers 8 \
  --gpus 1 \
  --early-stop-patience "$PATIENCE" \
  --gradient-clip-val 1.0 \
  --init-mode "$INIT" \
  "${CKPT_ARGS[@]}" \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "boxes-${SPLIT}-${INIT}-lr0.01-datascale-${TIER}"
