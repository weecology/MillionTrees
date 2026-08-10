#!/bin/bash
# TreeFormer point retrain at image_size=896, selecting checkpoints on val_loss
# instead of point_recall, and running the full 20 epochs.
#
# WHY THIS EXISTS: the v0.23 leaderboard point rows came from job 38750849, whose
# "fine-tuned" model was effectively the pretrained model plus one epoch. Both
# ModelCheckpoint and EarlyStopping monitored point_recall (mode=max), and on
# v0.23 point_recall saturates: it sat at ~0.93 +/- 0.005 of noise and hit its max
# at epoch 0 (out-of-distribution) / epoch 1 (within-distribution). So
# save_top_k=1 kept the epoch-0 weights, and EarlyStopping killed both runs at
# patience 10 (11 and 12 epochs of the requested 20).
#
# Meanwhile the run was still improving on every other axis, monotonically, right
# up to the epoch it was cut off (Comet 1e8bb9ae / b91436eb):
#   OOD  val_loss        0.4598 -> 0.4476   (down every epoch)
#        point_precision 0.1249 -> 0.1317
#        val_mae         374.3  -> 341.0
#   WD   val_loss        0.3941 -> 0.3662   (down every epoch)
#        point_precision 0.1681 -> 0.1858
#        val_mae         347.8  -> 319.5
# The original code comment justifying the recall monitor ("val_loss can drift
# upward while detections stay good") does not hold on v0.23 -- val_loss is
# monotone decreasing on both splits, and neither run ever overfit.
#
# FIX: --ckpt-monitor val_loss (mode=min, new flag on train.py) + --no-early-stop
# so the full 20 epochs run and the saved checkpoint reflects the whole run.
# Because val_loss never turned upward within 12 epochs, 20 may still be short of
# convergence -- read the tail of the Comet curve before deciding whether to
# extend further.
#
# Output goes to *_896_valloss dirs: a separate experiment that does NOT clobber
# the *_896_isolated checkpoints/results behind the current leaderboard rows.
#
# Runs in the frozen .venv-treeformer (DeepForest treeformer-training branch),
# called directly -- NOT `uv run`: the shared .venv is pinned to the polygon
# Mask R-CNN branch, which silently collapses TreeFormer point training
# (memory points-wrong-branch-collapse, ledger 36162735).
#SBATCH --job-name=mt_train_points_896_valloss
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=36:00:00
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
  --max-epochs 20 \
  --lr 2e-4 \
  --num-workers 12 \
  --gpus 3 \
  --ckpt-monitor val_loss \
  --no-early-stop \
  --output-dir "training/points/outputs/${SPLIT}_896_valloss" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-896-valloss" \
  --comet
