#!/bin/bash
# Stage 1 of the TreePoints weak-supervision ablation (points analogue of the
# box->box / box->polygon weak-supervision arms). Continue-pretrains the released
# TreeFormer point checkpoint (weecology/deepforest-tree-point) on AutoArborist
# (Beery et al. 2022 unsupervised, ~22k images) and exports an HF-format
# checkpoint that training/points/train.py --checkpoint consumes in stage 2.
#
# Array over both split schemes: AutoArborist is all-train for each, so stage 1
# trains on identical rows; the scheme only picks which split's supervised train
# the stage-1 val holdout is drawn from (never test, never validation).
#
# Runs in the frozen .venv-treeformer (DeepForest treeformer-training branch),
# called directly -- NOT `uv run` (memory: points-wrong-branch-collapse).
#SBATCH --job-name=mt_pretrain_points_aa
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

# Record real peak memory to /home/b.weinstein/logs/mem_ledger.csv on exit
# (scripts/slurm_mem_audit.py reads this to right-size --mem).
source scripts/slurm_mem_probe.sh

VENV=/blue/ewhite/b.weinstein/src/MillionTrees/.venv-treeformer
ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR=training/points/weak_supervision_outputs/pretrain

# Fail loud if the isolated env is not the treeformer-training branch.
"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

"$VENV/bin/python" training/points/pretrain_on_autoarborist.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --init-checkpoint weecology/deepforest-tree-point \
  --image-size 896 \
  --batch-size 16 \
  --max-epochs 10 \
  --lr 1e-5 \
  --num-workers 12 \
  --gpus 1 \
  --ckpt-monitor val_loss \
  --output-dir "$OUT_DIR" \
  --comet-name "points-${SPLIT}-autoarborist-pretrain-lr1e-5-896" \
  --comet

echo "=== stage 1 done: $OUT_DIR/$SPLIT/hf_checkpoint ==="
