#!/bin/bash
#SBATCH --job-name=mt_boxes_conv_pretrain_cotrain_v024
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# v0.24 re-run of the three-arm box->box weak-supervision ablation. Byte-identical
# to training/slurm/train_boxes_converged_pretrain_vs_cotrain.sbatch except:
#   - dataset version -> 0.24 (loader default), which repairs the OOD split
#   - ABLATION_DIR / PRETRAIN_DIR suffixed _v024 so v0.23 results are not clobbered
#   - comet names suffixed -v024
#
# THREE ARMS on one split (SPLIT env var, default within-distribution), all scored
# on that split's v0.24 test set:
#   0 coco               torchvision COCO RetinaNet, assert_coco_trunk enforced.
#   1 box_pretrained_full the converged v0.24 stage-1 RetinaNet (301 tensors, strict
#                        load) fine-tuned on the human boxes. Requires the split-matched
#                        export from pretrain_backbone_converged_v024.sbatch.
#   2 cotrain            one stage, COCO init, weak + human boxes together
#                        (--include-unsupervised --exclude-sources '*Young*').
#
# Hyperparameters identical across arms (lr 0.01, batch 32, clip 1.0, patience 10 on
# box_recall). Arm 2's epoch is longer, so at equal patience it gets more updates --
# train-each-regime-to-its-own-stopping-point, not matched compute.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLIT="${SPLIT:-within-distribution}"

PRETRAIN_DIR="training/weak_supervision/outputs/pretrain_converged_v024/lr1e-3-cosine-clip"
ABLATION_DIR="training/boxes/outputs/converged_ablation_v024"
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
              --comet-name "boxes-${SPLIT}-coco-lr0.01-converged-ctl-v024")
    ;;
  pretrained)
    if [ ! -f "$NETWORK" ]; then
      echo "ERROR: converged full-network export not found: $NETWORK" >&2
      echo "It comes from training/slurm/pretrain_backbone_converged_v024.sbatch." >&2
      echo "(box_backbone_*.pt is the trunk-only export and will be REJECTED here.)" >&2
      exit 1
    fi
    ARM_ARGS=(--init-mode box_pretrained_full
              --box-backbone-checkpoint "$NETWORK"
              --comet-name "boxes-${SPLIT}-box_pretrained_full-lr0.01-converged-v024")
    ;;
  cotrain)
    ARM_ARGS=(--init-mode coco
              --include-unsupervised
              --exclude-sources '*Young*'
              --comet-name "boxes-${SPLIT}-coco-lr0.01-cotrain-weak-v024")
    ;;
esac

srun uv run python training/boxes/train.py "${COMMON[@]}" "${ARM_ARGS[@]}"
