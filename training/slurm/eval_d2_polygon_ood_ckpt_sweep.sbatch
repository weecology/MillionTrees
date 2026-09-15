#!/bin/bash
#SBATCH --job-name=mt_d2_ckpt_sweep
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-25
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Offline checkpoint-trajectory sweep for the v0.24 Detectron2 polygon
# weak-supervision ablation, OUT-OF-DISTRIBUTION split. Inference only -- no
# retraining. Evaluates every retained checkpoint of the `coco` control and the
# `box_pretrained_full` (sequential) arm with the MillionTrees streaming polygon
# evaluator, so we can see the recall / mask-aware precision / F1 / AP40 / AP60
# curve over training and answer: does an early-stopped / pre-LR-decay control
# reach the co-training arm's F1 (0.734, recall 0.686) on its own?
#
#   STEPS = (36295, 46665) of MAX_ITER 51850; iters/epoch = 863 (6902 imgs / 8).
#   ckpts <= it 34519 are pre-decay (LR 0.01); 38834/43149 at LR 1e-3;
#   47464/51779/final at LR 1e-4.
#
# array 0-12  -> coco control checkpoints
# array 13-25 -> box_pretrained_full checkpoints
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLIT=out-of-distribution
ABLATION_DIR="$REPO/training/polygons/outputs/detectron2_ablation/$SPLIT"

CKPTS=(
  model_0004314.pth model_0008629.pth model_0012944.pth model_0017259.pth
  model_0021574.pth model_0025889.pth model_0030204.pth model_0034519.pth
  model_0038834.pth model_0043149.pth model_0047464.pth model_0051779.pth
  model_final.pth
)
LABELS=(ep005 ep010 ep015 ep020 ep025 ep030 ep035 ep040 ep045 ep050 ep055 ep060 epfinal)

if [ "$SLURM_ARRAY_TASK_ID" -lt 13 ]; then
  ARM=coco
  IDX=$SLURM_ARRAY_TASK_ID
else
  ARM=box_pretrained_full
  IDX=$((SLURM_ARRAY_TASK_ID - 13))
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

# --init-mode coco: the checkpoint carries a complete model state, so no stage-1
# merge / weak-set loading is needed regardless of which arm produced it.
# Fresh --output-dir (no model_final.pth in it) so --weights is the model that loads.
python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --init-mode coco \
  --eval-only \
  --eval-split test \
  --weights "$SRC_CKPT" \
  --batch-size 8 \
  --num-workers 8 \
  --image-size 448 \
  --output-dir "$OUT_DIR" \
  --save-predictions "$OUT_DIR/preds.pkl"

echo "=== done: $OUT_DIR ==="
