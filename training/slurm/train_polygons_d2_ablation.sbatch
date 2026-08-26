#!/bin/bash
#SBATCH --job-name=mt_poly_d2_ablation
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

# STAGE 2: the three-arm weak-supervision ablation on the NATIVE Detectron2 Mask R-CNN,
# the stack behind the leaderboard polygon rows. One split per submit (SPLIT env var).
#
#   0  coco                 Tree-naive control. Backbone/FPN verified tensor-for-tensor
#                           against the released COCO Mask R-CNN, because DeepForest's
#                           config default silently substituted a NEON-trained checkpoint
#                           in two earlier ablations and invalidated them.
#   1  box_pretrained_full  Sequential: the whole stage-1 network from
#                           pretrain_detectron2.py, merged onto COCO so the mask head
#                           stays COCO-init. Refuses to run if the backbone came out
#                           byte-identical to COCO, i.e. a control in disguise.
#   2  cotrain              The weak Weinstein tiles mixed into the polygon training set
#                           in ONE stage. Box-only tiles contribute RPN and box losses;
#                           MaskAwareROIHeads holds them out of the mask loss, so the mask
#                           head never learns to predict rectangles.
#
# MATCHED COMPUTE, unlike the box round. --max-iter is pinned identically across arms:
# the co-training arm's epoch is ~5x longer (9,317 polygon tiles vs 51,008 mixed), so a
# shared "50 epochs" would hand it 5x the updates and confound the weak labels with the
# budget. The box round had to carry that as a caveat; here it is designed out.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT="${SPLIT:-within-distribution}"

# 50 epochs' worth of the CONTROL arm's data, i.e. the recipe behind the published
# Detectron2 polygon rows: WD 9,317 imgs / 8 = 1,165 it/ep; OOD 8,290 / 8 = 1,037 it/ep.
if [ "$SPLIT" = "out-of-distribution" ]; then
  DEFAULT_MAX_ITER=51850
else
  DEFAULT_MAX_ITER=58250
fi
MAX_ITER="${MAX_ITER:-$DEFAULT_MAX_ITER}"

PRETRAIN_DIR="$REPO/training/weak_supervision/outputs/pretrain_detectron2/$SPLIT"
STAGE1_EXPORT="$PRETRAIN_DIR/d2_network_${SPLIT}.pkl"
ABLATION_DIR="$REPO/training/polygons/outputs/detectron2_ablation/$SPLIT"

case "$SLURM_ARRAY_TASK_ID" in
  0) ARM=coco                ;;
  1) ARM=box_pretrained_full ;;
  2) ARM=cotrain             ;;
  *) echo "unknown array index $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
OUT_DIR="$ABLATION_DIR/$ARM"
mkdir -p "$OUT_DIR"

COMMON=(
  --root-dir "$ROOT_DIR"
  --split-scheme "$SPLIT"
  --batch-size 8
  --max-iter "$MAX_ITER"
  --lr 0.01
  --grad-clip 1.0
  --num-workers 16
  --image-size 448
  --output-dir "$OUT_DIR"
  --comet
)

case "$ARM" in
  coco)
    ARM_ARGS=(--init-mode coco
              --comet-name "polygons-${SPLIT}-coco-lr0.01-d2-ablation-ctl")
    ;;
  box_pretrained_full)
    if [[ ! -f "$STAGE1_EXPORT" ]]; then
      echo "ERROR: stage-1 export not found: $STAGE1_EXPORT" >&2
      echo "Run training/slurm/pretrain_detectron2.sbatch for SPLIT=$SPLIT first." >&2
      exit 1
    fi
    ARM_ARGS=(--init-mode box_pretrained_full
              --weights "$STAGE1_EXPORT"
              --comet-name "polygons-${SPLIT}-box_pretrained_full-lr0.01-d2-ablation")
    ;;
  cotrain)
    # Same weak set stage 1 sees, so the only difference between arms 1 and 2 is WHEN the
    # weak labels are used, not WHICH ones.
    ARM_ARGS=(--init-mode cotrain
              --boxes-dataset TreeBoxes_v0.23
              --weak-sources '*unsupervised*'
              --comet-name "polygons-${SPLIT}-coco-lr0.01-d2-cotrain-weak")
    ;;
esac

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Block-buffered stdout makes a running job unreadable; the ledger workflow reads .out
# while the job is still going.
export PYTHONUNBUFFERED=1

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== D2 polygon ablation: split=$SPLIT arm=$ARM max_iter=$MAX_ITER ==="
python "$REPO/training/polygons/train_detectron2.py" "${COMMON[@]}" "${ARM_ARGS[@]}"
echo "=== D2 polygon ablation done: $OUT_DIR ==="
