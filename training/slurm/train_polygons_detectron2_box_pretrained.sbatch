#!/bin/bash
#SBATCH --job-name=mt_poly_d2_boxpre
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Table 6 box-pretrained arm on the NATIVE Detectron2 driver.
#
# The COCO arm is the existing detectron2 run (job 36100956,
# outputs/detectron2/<split>). This is its twin with an identical recipe
# (batch 8, 50 ep, lr 0.01, image_size 448) whose only difference is the init:
# the ResNet trunk is the unsupervised box-pretrained backbone (v0.20) merged
# into the COCO Mask R-CNN via convert_box_backbone_to_d2.py -- FPN/RPN/ROI heads
# stay COCO-pretrained, exactly as the DeepForest box arm keeps torchvision-COCO
# for the non-backbone weights. Table 6 = these two arms x {within, OOD}.
#
# Runs under the CanopyRS uv venv (from-source Detectron2); array index selects
# the split (0 within-distribution, 1 out-of-distribution).
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="$REPO/training/polygons/outputs/detectron2_box_pretrained/$SPLIT"
WEIGHTS="$REPO/training/polygons/detectron2_assets/box_pretrained/box_backbone_${SPLIT}.pkl"
mkdir -p "$OUT_DIR"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: merged box-pretrained pkl not found: $WEIGHTS" >&2
  echo "Run: python training/polygons/convert_box_backbone_to_d2.py --box-backbone \\" >&2
  echo "  training/weak_supervision/outputs/pretrain_v020/$SPLIT/box_backbone_$SPLIT.pt --out $WEIGHTS" >&2
  exit 1
fi

# Detectron2 (CanopyRS venv) CUDA ops are a fat binary for L4 (sm_89) + B200 (sm_100).
module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "=== Detectron2 polygon Table-6 box-pretrained arm: split=$SPLIT ==="
echo "=== weights=$WEIGHTS ==="

python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs 50 \
  --lr 0.01 \
  --num-workers 16 \
  --image-size 448 \
  --weights "$WEIGHTS" \
  --init-mode box-pretrained \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-box-pretrained-lr0.01-detectron2"

echo "=== Detectron2 polygon Table-6 box-pretrained arm done: $OUT_DIR ==="
