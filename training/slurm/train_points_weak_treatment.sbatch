#!/bin/bash
# Stage 2, TREATMENT arm of the TreePoints weak-supervision ablation: fine-tune
# on the supervised MillionTrees points train split starting from the
# AutoArborist-pretrained checkpoint exported by
# training/points/pretrain_on_autoarborist.py (stage 1).
#
# Byte-for-byte the same call as train_points_weak_control.sbatch except
# --checkpoint points at the stage-1 HF export. SPLIT is passed via
# `sbatch --export=ALL,SPLIT=...` and this job should carry an afterok
# dependency on the matching pretrain_points_autoarborist array task
# (see training/weak_supervision/submit_points_weak_supervision.sh).
#
# Runs in the frozen .venv-treeformer, called directly (memory: points-wrong-branch-collapse).
#SBATCH --job-name=mt_points_weak_treat
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
: "${SPLIT:?set SPLIT via --export=ALL,SPLIT=within-distribution|out-of-distribution}"
PRETRAIN_DIR="training/points/weak_supervision_outputs/pretrain/${SPLIT}/hf_checkpoint"
OUT_DIR="training/points/weak_supervision_outputs/treatment/${SPLIT}"

"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

if [ ! -f "$PRETRAIN_DIR/config.json" ]; then
  echo "ERROR: stage-1 HF checkpoint missing at $PRETRAIN_DIR (did pretrain job succeed?)" >&2
  exit 1
fi

srun "$VENV/bin/python" training/points/train.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --checkpoint "$PRETRAIN_DIR" \
  --image-size 896 \
  --batch-size 16 \
  --max-epochs 20 \
  --lr 2e-4 \
  --num-workers 12 \
  --gpus 3 \
  --ckpt-monitor val_loss \
  --output-dir "$OUT_DIR" \
  --comet-name "points-${SPLIT}-autoarborist-lr2e-4-896-weaktreat" \
  --comet

echo "=== treatment arm done: $OUT_DIR/results_${SPLIT}.txt ==="
