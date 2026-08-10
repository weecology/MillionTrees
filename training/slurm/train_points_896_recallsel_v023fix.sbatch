#!/bin/bash
# TreeFormer point retrain at image_size=896 with the ORIGINAL point_recall
# checkpoint selection + early stopping, re-run on the REPAIRED v0.23 data.
#
# WHY THIS EXISTS: this is the control arm for the checkpoint-selection question.
# Job 38826342 was supposed to isolate val_loss vs point_recall selection, but the
# dataset changed underneath it -- the in-place TreePoints repackage (38827912,
# 08-07 02:33) restored 187 Amirkolaee et al. 2023 images / 32,452 annotations
# between the recall-selected run (38750849, old data) and the val_loss run
# (38826334, fixed data). So the observed F1 drop (within-distribution 0.743 ->
# 0.728, out-of-distribution 0.710 -> 0.659, driven by OOD precision 0.635 ->
# 0.538) cannot be attributed to selection: data and selection both moved.
#
# This run reproduces the OLD selection config exactly -- default point_recall
# monitor (mode=max), early stopping on, 20 epochs -- on the FIXED data, so
# comparing it against 38826334 varies selection alone.
#
# Config is otherwise identical to train_points_896_isolated.sbatch. Output goes
# to *_896_recallsel_v023fix dirs so the _896_isolated checkpoints and results
# from the old-data run stay intact as the historical record.
#
# Runs in the frozen .venv-treeformer (DeepForest treeformer-training branch),
# called directly -- NOT `uv run`: the shared .venv is pinned to the polygon
# Mask R-CNN branch, which silently collapses TreeFormer point training
# (memory points-wrong-branch-collapse, ledger 36162735).
#SBATCH --job-name=mt_train_points_896_recallsel
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
  --output-dir "training/points/outputs/${SPLIT}_896_recallsel_v023fix" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-896-recallsel-v023fix" \
  --comet
