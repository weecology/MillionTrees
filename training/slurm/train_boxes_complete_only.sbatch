#!/bin/bash
#SBATCH --job-name=mt_train_boxes_completeonly
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=80GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3

# Paired counterpart to train_boxes_lr_sweep.sbatch: identical 2x2 split x lr
# matrix, but trains ONLY on complete=True (exhaustively annotated) sources via
# --train-complete-only. The filter applies to the TRAIN split only; the test
# set is unchanged, so these runs are directly comparable to the full-train
# lr-sweep baselines (35574489_{0..3}).
#   idx 0: within-distribution  lr=0.005
#   idx 1: within-distribution  lr=0.001
#   idx 2: out-of-distribution  lr=0.005
#   idx 3: out-of-distribution  lr=0.001
# Dataset is loaded from ROOT_DIR/TreeBoxes_v{version} (version from TreeBoxes._versions_dict).
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
# Reclaim reserved-but-unallocated fragmentation; batch 32 is marginal on 22GiB L4.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/boxes/outputs

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution within-distribution out-of-distribution out-of-distribution)
LRS=(0.005 0.001 0.005 0.001)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

# Out-of-distribution needs more time to find a good optimum.
if [ "$SPLIT" = "out-of-distribution" ]; then
  EARLY_STOP_PATIENCE=15
else
  EARLY_STOP_PATIENCE=10
fi

srun uv run python training/boxes/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 32 \
  --max-epochs 200 \
  --lr "$LR" \
  --num-workers 8 \
  --gpus 3 \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --remove-incomplete \
  --output-dir "training/boxes/outputs/${SPLIT}_lr${LR}_completeonly" \
  --comet-name "boxes-${SPLIT}-pretrained-lr${LR}-completeonly" \
  --comet
