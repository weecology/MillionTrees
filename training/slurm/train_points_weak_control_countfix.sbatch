#!/bin/bash
# COUNT-LOSS-FIX probe of the TreePoints weak-supervision ablation, CONTROL arm.
#
# Identical to train_points_weak_control.sbatch except it adds
#   --loss-preset pretrain --no-early-stop
# i.e. the exact recipe behind manuscript Doc Table 2 (train_points_896_countfix).
# The plain weak-supervision ablation used the leaderboard recipe WITHOUT the
# count-loss fix, so its absolute numbers are not comparable to Table 2. This
# probe re-runs the decisive OOD control/treatment pair under the count-fix recipe
# to check whether the +0.050 validation-F1 AutoArborist gain survives.
#
# Single split (out-of-distribution) -- the split where the plain-recipe
# validation gain was largest and the manuscript generalisation claim lives.
# Reuses the existing stage-1 HF export (stage 1 already used point_pretrain.yaml).
#
# Runs in the frozen .venv-treeformer, called directly (memory: points-wrong-branch-collapse).
#SBATCH --job-name=mt_points_weak_ctrl_cfix
#SBATCH --output=/home/b.weinstein/logs/%x_%A.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=480GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

VENV=/blue/ewhite/b.weinstein/src/MillionTrees/.venv-treeformer
ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=out-of-distribution
OUT_DIR="training/points/weak_supervision_outputs/control_countfix/${SPLIT}"

"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

srun "$VENV/bin/python" training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --checkpoint weecology/deepforest-tree-point \
  --image-size 896 \
  --batch-size 16 \
  --max-epochs 20 \
  --lr 2e-4 \
  --num-workers 12 \
  --gpus 3 \
  --loss-preset pretrain \
  --ckpt-monitor val_loss \
  --no-early-stop \
  --output-dir "$OUT_DIR" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-896-weakctrl-countfix" \
  --comet

echo "=== control countfix arm done: $OUT_DIR/results_${SPLIT}.txt ==="
