#!/bin/bash
#SBATCH --job-name=mt_d2_ckpt_sweep_wd
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-19
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Offline checkpoint-trajectory sweep for the v0.24 Detectron2 polygon
# weak-supervision ablation, WITHIN-DISTRIBUTION split. Inference only -- no
# retraining. Evaluates every retained 5-epoch checkpoint of the `coco` control
# and the `box_pretrained_full` (sequential) arm with the MillionTrees streaming
# polygon evaluator, so we can see the recall / mask-aware precision / F1 / AP40 /
# AP60 curve over training and select each Table 6 arm on peak (val-proxy) F1
# rather than the last iterate. The OOD sweep (job 40990287) showed the OOD
# control peaks at ep~10 (F1 0.760) then overfits monotonically to the
# last-iterate 0.538 -- this run checks whether the WD arms do the same.
#
#   [cfg] 9408 imgs | 1176 iters/epoch x 50 ep = 58250 iters | lr 0.01
#         steps (40775, 52425): ckpts <= 40774 are pre-decay (LR 0.01);
#         41159/47039 at LR 1e-3; 52919/final at LR 1e-4.
#
# array 0-9   -> coco control checkpoints (ep05..ep45, epfinal)
# array 10-19 -> box_pretrained_full checkpoints (ep05..ep45, epfinal)
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=within-distribution
ABLATION_DIR="$REPO/training/polygons/outputs/detectron2_ablation/$SPLIT"

CKPTS=(
  model_0005879.pth model_0011759.pth model_0017639.pth model_0023519.pth
  model_0029399.pth model_0035279.pth model_0041159.pth model_0047039.pth
  model_0052919.pth model_final.pth
)
LABELS=(ep005 ep010 ep015 ep020 ep025 ep030 ep035 ep040 ep045 epfinal)

if [ "$SLURM_ARRAY_TASK_ID" -lt 10 ]; then
  ARM=coco
  IDX=$SLURM_ARRAY_TASK_ID
else
  ARM=box_pretrained_full
  IDX=$((SLURM_ARRAY_TASK_ID - 10))
fi
CKPT="${CKPTS[$IDX]}"
LABEL="${LABELS[$IDX]}"

SRC_CKPT="$ABLATION_DIR/$ARM/$CKPT"
OUT_DIR="$ABLATION_DIR/${ARM}_ckpt_sweep/$LABEL"
mkdir -p "$OUT_DIR"

if [[ ! -f "$SRC_CKPT" ]]; then
  echo "ERROR: checkpoint not found: $SRC_CKPT" >&2
  exit 1
fi

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== D2 polygon ckpt sweep: split=$SPLIT arm=$ARM ckpt=$CKPT label=$LABEL ==="

# Score every checkpoint on BOTH the WD test set (for the reported Table 6
# numbers of the selected ckpt) and the held-out Allen+Frey validation set
# (the val-F1 proxy the ledger requires for checkpoint selection). The
# validation set is tiny so the second pass is cheap.
#
# --init-mode coco: the checkpoint carries a complete model state, so no stage-1
# merge / weak-set loading is needed regardless of which arm produced it.
# Fresh --output-dir (no model_final.pth in it) so --weights is the model that loads.
for EVAL_SPLIT in test validation; do
  echo "--- eval_split=$EVAL_SPLIT ---"
  python "$REPO/training/polygons/train_detectron2.py" \
    --root-dir "$ROOT_DIR" \
    --split-scheme "$SPLIT" \
    --init-mode coco \
    --eval-only \
    --eval-split "$EVAL_SPLIT" \
    --weights "$SRC_CKPT" \
    --batch-size 8 \
    --num-workers 8 \
    --image-size 448 \
    --output-dir "$OUT_DIR" \
    --save-predictions "$OUT_DIR/preds_${EVAL_SPLIT}.pkl"
done

echo "=== done: $OUT_DIR ==="
