#!/bin/bash
# TreeFormer point retrain at image_size=896, val_loss-selected, 40 epochs.
#
# WHY THIS EXISTS: job 38826334 ran the same config for 20 epochs on the repaired
# v0.23 data and did NOT converge -- save_top_k=1 kept epoch 19 of 20 on BOTH
# splits (within-distribution val_loss=0.3620, out-of-distribution val_loss=0.4902),
# i.e. val_loss was still at its minimum on the final epoch. Its own header note
# anticipated this: "Because val_loss never turned upward within 12 epochs, 20 may
# still be short of convergence -- read the tail of the Comet curve before deciding
# whether to extend further." This is that extension.
#
# Only --max-epochs changes vs train_points_896_valloss.sbatch. Output goes to
# *_896_valloss40 dirs so the 20-epoch checkpoints and results stay intact for
# comparison.
#
# Runs in the frozen .venv-treeformer (DeepForest treeformer-training branch),
# called directly -- NOT `uv run`: the shared .venv is pinned to the polygon
# Mask R-CNN branch, which silently collapses TreeFormer point training
# (memory points-wrong-branch-collapse, ledger 36162735).
#SBATCH --job-name=mt_train_points_896_valloss40
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
mkdir -p training/points/outputs

VENV=/blue/ewhite/b.weinstein/src/MillionTrees/.venv-treeformer
ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

# Fail loud if the isolated env is not the treeformer-training branch.
"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

srun "$VENV/bin/python" training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --image-size 896 \
  --batch-size 16 \
  --max-epochs 40 \
  --lr 2e-4 \
  --num-workers 12 \
  --gpus 3 \
  --ckpt-monitor val_loss \
  --no-early-stop \
  --output-dir "training/points/outputs/${SPLIT}_896_valloss40" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-896-valloss-40ep" \
  --comet
