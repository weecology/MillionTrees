#!/bin/bash
#SBATCH --job-name=mt_polygons_maskrcnn_pretrained
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# STAGE 2 of the Mask R-CNN weak-supervision arm. One split per submission; the
# array's two tasks are the TREATMENT and its PAIRED CONTROL, submitted together
# on purpose.
#
#   0 coco                    torchvision COCO Mask R-CNN v2, assert_coco_trunk()
#                             verifies all 366 backbone/FPN tensors before training.
#   1 box_pretrained_maskrcnn 404 of 432 tensors from the stage-1 Mask R-CNN --
#                             backbone + FPN + RPN + ROI box head. Only the
#                             28-tensor mask head stays at COCO.
#   2 box_pretrained (trunk)  The SAME stage-1 network, trunk only. This arm exists
#                             to keep the result attributable. Arm 1 changes two
#                             things at once versus the published Table 6 row --
#                             more tensors transfer, AND stage 1 is now a Mask R-CNN
#                             rather than a RetinaNet. Arm 2 holds the architecture
#                             fixed and varies only the transfer depth, so
#                             (2 - 0) isolates the architecture change and
#                             (1 - 2) isolates the extra 86 tensors. Without it a
#                             win has two live explanations and needs this run
#                             afterwards anyway.
#                             Note its trunk carries 318 tensors, not the 265 the
#                             RetinaNet export managed, so it is not identical to
#                             the old arm either.
#
# WHY THE CONTROL IS RE-RUN RATHER THAN REUSED. The box-side ablation ran its
# COCO arm twice under an identical config and got AP40 0.476 vs 0.450 and
# mask-aware precision 0.633 vs 0.588. That 0.026 / 0.045 spread is run-to-run
# variance, and it is LARGER than the pretraining effect the experiment is trying
# to measure. Every delta in notes/weak_supervision_pretraining_table.md was
# computed against whichever control replicate happened to be on disk, which is
# how a consistent 13-of-16-source precision gain got written up as a null. The
# polygon side has no replicate at all and Mask R-CNN overfits these splits by
# epoch 2-5, so its spread should be wider still. An unpaired control here would
# make this run uninterpretable no matter which way it came out.
#
# WHAT THIS ARM TESTS, precisely: whether the polygon null survives once the
# transfer is no longer truncated. The existing box_pretrained arm moves 265
# tensors and is a null WD / a degradation OOD; this moves 339 more, including
# every layer that encodes "this is a tree and this is its extent". On the box
# side that same change -- trunk-only to whole-network -- moved AP40 from -0.016
# to exactly the COCO baseline.
#
# Read the two results_<split>.txt PER SOURCE, not just the macro averages. The
# WD macro AP40 delta on the previous arm was -0.020 and was entirely one
# 18-image source (Safonova 0.960 -> 0.051); excluding it the delta is +0.017.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Unbuffered stdout. Without this, python's stdout is block-buffered when
# redirected to a file while stderr is not, so the .out log stays EMPTY for
# hours on a long run while .err fills -- which silently defeats the
# 'check the .out to confirm it is running' step in CLAUDE.md section 1.
export PYTHONUNBUFFERED=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLIT="${SPLIT:-within-distribution}"
TAG=maskrcnn-lr1e-3-cosine-clip
STAGE1_DIR="training/weak_supervision/outputs/pretrain_maskrcnn/$TAG/$SPLIT"
NETWORK="$STAGE1_DIR/maskrcnn_transferable_${SPLIT}.pt"
TRUNK="$STAGE1_DIR/box_backbone_${SPLIT}.pt"

case "$SLURM_ARRAY_TASK_ID" in
  0) ARM=coco
     OUT_DIR="training/weak_supervision/outputs/polygon_maskrcnn_ablation/$SPLIT/coco"
     ARM_ARGS=(--init-mode coco
               --comet-name "polygons-${SPLIT}-coco-lr0.01-maskrcnn-ablation-ctl") ;;
  1) ARM=pretrained
     OUT_DIR="training/weak_supervision/outputs/polygon_maskrcnn_ablation/$SPLIT/box_pretrained_maskrcnn"
     ARM_ARGS=(--init-mode box_pretrained_maskrcnn
               --box-backbone-checkpoint "$NETWORK"
               --comet-name "polygons-${SPLIT}-box_pretrained_maskrcnn-lr0.01") ;;
  2) ARM=trunkonly
     OUT_DIR="training/weak_supervision/outputs/polygon_maskrcnn_ablation/$SPLIT/box_pretrained_trunk"
     ARM_ARGS=(--init-mode box_pretrained
               --box-backbone-checkpoint "$TRUNK"
               --comet-name "polygons-${SPLIT}-box_pretrained_trunk-lr0.01-maskrcnn-stage1") ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
mkdir -p "$OUT_DIR"

if [ "$ARM" = "pretrained" ] && [ ! -f "$NETWORK" ]; then
  echo "ERROR: stage-1 Mask R-CNN export not found: $NETWORK" >&2
  echo "It comes from training/slurm/pretrain_maskrcnn_for_polygons.sbatch." >&2
  echo "(box_backbone_*.pt is the trunk-only export; train.py REJECTS it here.)" >&2
  exit 1
fi
if [ "$ARM" = "trunkonly" ] && [ ! -f "$TRUNK" ]; then
  echo "ERROR: stage-1 trunk export not found: $TRUNK" >&2
  exit 1
fi

# Hyperparameters identical across both arms so init is the only variable.
srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --max-epochs 100 \
  --lr 0.01 \
  --num-workers 16 \
  --image-size 448 \
  --data-scope subset \
  --warmup-epochs 2 \
  --output-dir "$OUT_DIR" \
  --comet \
  "${ARM_ARGS[@]}"
