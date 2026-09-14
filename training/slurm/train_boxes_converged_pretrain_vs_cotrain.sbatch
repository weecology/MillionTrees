#!/bin/bash
#SBATCH --job-name=mt_boxes_conv_pretrain_cotrain
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Does the weak-label pretraining help the SAME-GEOMETRY box task, now that stage 1
# actually converges -- and is doing it all at once better than doing it in two stages?
#
# Every previous answer to the first question was compromised by a stage-1 network that
# was either truncated on transfer (39129721, trunk only), diverging (39879762 task 1,
# no gradient clipping), or checkpoint-selected on a loss that does not track detection
# (both sweep arms exported a val_loss-best epoch; lr1e-3's was recall 0.003 / AP40 0.001).
# 39974598 fixed all three -- --grad-clip 1.0, --ckpt-monitor box_recall --ckpt-mode max --
# and produced the first stage-1 run whose loss falls cleanly (train_loss 0.87, peak
# box_recall 0.274 vs 0.237 for the diverged sweep arm). Its 301-tensor export is what
# arm 1 transfers.
#
# THREE ARMS, all on one split (SPLIT env var, default within-distribution), all scored
# on that split's test set (3,023 images within-distribution):
#
#   0 coco               torchvision COCO RetinaNet, no tree exposure of any kind.
#                        assert_coco_trunk() enforces this -- DeepForest's config default
#                        is weecology/deepforest-tree and has silently contaminated two
#                        earlier "clean" baselines (38840964, 39089781).
#                        Re-run rather than reusing 39129720 (AP40 0.476) so all three
#                        arms share one code state and one dataset read.
#
#   1 box_pretrained_full  The SEQUENTIAL route: the whole converged stage-1 RetinaNet
#                        (trunk + FPN + both heads, 301 tensors, strict load) fine-tuned
#                        on the human boxes. Stage 1 itself starts from COCO, so arm 0 is
#                        exactly "what pretraining started from, fine-tuned" -- the paired
#                        control.
#
#   2 cotrain            The SIMULTANEOUS route: one stage, COCO init, weak and human
#                        boxes optimized together (--include-unsupervised). 55,229 train
#                        images = 13,538 supervised + 41,691 Weinstein et al. 2018
#                        pseudo-labelled; 7.2M boxes of which 6.4M are weak. Weak rows are
#                        train-only, so test is untouched and the AP40 is comparable.
#
# --exclude-sources '*Young*' on arm 2: the full v0.23 release carries a SECOND weak
# source (Young et al. 2025 weak supervised, 11,739 images / 99,626 boxes) that stage-1
# pretraining never saw -- it trains on include_sources=['*unsupervised*'], i.e. Weinstein
# only. Letting it in would confound "co-training vs pretraining" with "more weak data",
# so it is held out and the two routes see the identical weak set.
#
# Hyperparameters are identical across arms (lr 0.01, batch 32, clip 1.0, patience 10 on
# box_recall) so init/regime is the only variable. Note that arm 2's epoch is 4.1x longer
# (1,726 vs 423 optimizer steps), so at equal patience it also gets more updates before
# early stopping -- that is the intended "train each regime to its own stopping point"
# comparison, not a matched-compute one.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

# SPLIT is overridable so the same three arms can be run on the out-of-distribution
# split (39998524 was the within-distribution round). The default is unchanged and the
# WD output paths stay byte-identical; a non-default split gets its own subdirectory so
# it cannot clobber the WD checkpoints. The stage-1 network is split-specific too --
# arm 1 on OOD requires the OOD export from pretrain_backbone_converged.sbatch, which
# must never be the WD one (the WD stage 1 has seen imagery that is OOD test data).
SPLIT="${SPLIT:-within-distribution}"

PRETRAIN_DIR="training/weak_supervision/outputs/pretrain_converged/lr1e-3-cosine-clip"
ABLATION_DIR="training/boxes/outputs/converged_ablation"
if [ "$SPLIT" != "within-distribution" ]; then
  PRETRAIN_DIR="$PRETRAIN_DIR/$SPLIT"
  ABLATION_DIR="$ABLATION_DIR/$SPLIT"
fi
NETWORK="$PRETRAIN_DIR/box_network_${SPLIT}.pt"

case "$SLURM_ARRAY_TASK_ID" in
  0) ARM=coco       ; OUT_DIR="$ABLATION_DIR/coco" ;;
  1) ARM=pretrained ; OUT_DIR="$ABLATION_DIR/box_pretrained_full_converged" ;;
  2) ARM=cotrain    ; OUT_DIR="$ABLATION_DIR/cotrain_weak" ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
mkdir -p "$OUT_DIR"

COMMON=(
  --root-dir /orange/ewhite/web/public/MillionTrees
  --split-scheme "$SPLIT"
  --batch-size 32
  --max-epochs 200
  --lr 0.01
  --num-workers 16
  --gpus 1
  --early-stop-patience 10
  --gradient-clip-val 1.0
  --output-dir "$OUT_DIR"
  --comet
)

case "$ARM" in
  coco)
    ARM_ARGS=(--init-mode coco
              --comet-name "boxes-${SPLIT}-coco-lr0.01-converged-ctl")
    ;;
  pretrained)
    if [ ! -f "$NETWORK" ]; then
      echo "ERROR: converged full-network export not found: $NETWORK" >&2
      echo "It comes from training/slurm/pretrain_backbone_converged.sbatch (job 39974598)." >&2
      echo "(box_backbone_*.pt is the trunk-only export and will be REJECTED here.)" >&2
      exit 1
    fi
    ARM_ARGS=(--init-mode box_pretrained_full
              --box-backbone-checkpoint "$NETWORK"
              --comet-name "boxes-${SPLIT}-box_pretrained_full-lr0.01-converged")
    ;;
  cotrain)
    ARM_ARGS=(--init-mode coco
              --include-unsupervised
              --exclude-sources '*Young*'
              --comet-name "boxes-${SPLIT}-coco-lr0.01-cotrain-weak")
    ;;
esac

srun uv run python training/boxes/train.py "${COMMON[@]}" "${ARM_ARGS[@]}"
