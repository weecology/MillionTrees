#!/bin/bash
# Stage 2, CONTROL arm of the TreePoints weak-supervision ablation: fine-tune on
# the supervised MillionTrees points train split starting DIRECTLY from the
# released TreeFormer checkpoint (weecology/deepforest-tree-point). No
# AutoArborist pretraining stage.
#
# Identical recipe to the leaderboard points run (train_points_896_isolated:
# pretrained init, lr 2e-4, 20 epochs, image 896). The treatment arm
# (train_points_weak_treatment.sbatch) is byte-for-byte the same call with a
# different --checkpoint. train.py evaluates the selected checkpoint on the
# MillionTrees points TEST split at the end; the VAL split is scored separately
# by eval_validation_points-style follow-up.
#
# Runs in the frozen .venv-treeformer, called directly (memory: points-wrong-branch-collapse).
#SBATCH --job-name=mt_points_weak_control
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
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

VENV=/blue/ewhite/b.weinstein/src/MillionTrees/.venv-treeformer
ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/points/weak_supervision_outputs/control/${SPLIT}"

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
  --ckpt-monitor val_loss \
  --output-dir "$OUT_DIR" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-896-weakctrl" \
  --comet

echo "=== control arm done: $OUT_DIR/results_${SPLIT}.txt ==="
