#!/bin/bash
#SBATCH --job-name=mt_polygons_coco_v023_cleaninit
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

# Control arm of the CORRECTED polygon (Table 6) unsupervised-pretraining ablation,
# v0.23 / AP40.
#
# The predecessor (train_polygons_coco.sbatch, job 38750868) was labelled the "COCO
# baseline" and was not one. deepforest_polygon.yaml sets no model.name, so the base
# config default 'weecology/deepforest-tree' applied; create_model() therefore took the
# load_model() branch into MaskRCNN.from_pretrained(), which builds a cold-start shell
# and then loads a *RetinaNet* checkpoint whose keys never match the Mask R-CNN. Nothing
# landed on the backbone. Measured on the old config: 0/318 backbone.body tensors equal
# torchvision COCO_V1 and all 53 BatchNorm running_mean are still exactly 0, and two
# fresh builds disagree on all 53 conv weights -- i.e. the ResNet-50 was RANDOMLY
# INITIALIZED. Note this differs from the box-side bug: the box arms were contaminated
# with NEON weights, these polygon arms had no pretraining at all, which INFLATES the
# apparent benefit of the box-pretrained trunk.
#
# training/polygons/train.py now forces model.name=None in build_config() (routing to
# MaskRCNN(backbone_weights="COCO_V1")) and calls assert_coco_trunk() to verify all 366
# backbone+FPN tensors against torchvision maskrcnn_resnet50_fpn_v2 COCO_V1 before
# training. Same recipe otherwise; new output dir keeps 38750868 reproducible.
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/weak_supervision/outputs/polygon_coco_v023_cleaninit/$SPLIT"
mkdir -p "$OUT_DIR"

srun uv run python training/polygons/train.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --gpus 1 \
  --batch-size 16 \
  --max-epochs 100 \
  --lr 0.01 \
  --num-workers 16 \
  --image-size 448 \
  --init-mode coco \
  --data-scope subset \
  --output-dir "$OUT_DIR" \
  --comet \
  --comet-name "polygons-${SPLIT}-coco-lr0.01-v023-cleaninit"
