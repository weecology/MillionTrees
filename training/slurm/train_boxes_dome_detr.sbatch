#!/bin/bash
#SBATCH --job-name=mt_train_boxes_dome
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=l4:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2

# Dataset is loaded from ROOT_DIR/TreeBoxes_v{version} (version from TreeBoxes._versions_dict).
# Dome-DETR needs its own isolated Python environment (build_env.sh) due to compiled CUDA ops.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /blue/ewhite/b.weinstein/src/MillionTrees-dome-detr
mkdir -p training/boxes_dome_detr/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# Activate the isolated Dome-DETR environment
source training/boxes_dome_detr/.venv/bin/activate

# Out-of-distribution may need more epochs to converge
if [ "$SPLIT" = "out-of-distribution" ]; then
  MAX_EPOCHS=100
  LR=0.001
else
  MAX_EPOCHS=80
  LR=0.001
fi

python training/boxes_dome_detr/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --max-epochs "$MAX_EPOCHS" \
  --lr "$LR" \
  --num-workers 8 \
  --gpus 2 \
  --model-size m \
  --init-mode pretrained \
  --output-dir "training/boxes_dome_detr/outputs/$SPLIT" \
  --comet
