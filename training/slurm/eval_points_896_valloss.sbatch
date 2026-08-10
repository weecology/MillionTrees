#!/bin/bash
#SBATCH --job-name=mt_eval_points_896_valloss
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Test-split eval of the val_loss-selected TreeFormer point checkpoints from
# train_points_896_valloss.sbatch. Everything except checkpoint selection matches
# eval_points_896_isolated_test.sbatch (image_size 896, score_thresh 0.10), so the
# results are directly comparable to the current leaderboard point rows and
# isolate the effect of the selection change.
#
# NOTE the checkpoint glob differs from the *_896_isolated eval scripts: those
# filenames carry point_recall= and are sorted descending; these carry val_loss=
# and must be sorted ASCENDING (lower is better).
#
# Runs in the frozen .venv-treeformer (DeepForest treeformer-training branch),
# called directly -- NOT `uv run` (memory points-wrong-branch-collapse).
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

VENV=/blue/ewhite/b.weinstein/src/MillionTrees/.venv-treeformer
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
CKPT_DIR="training/points/outputs/${SPLIT}_896_valloss/checkpoints"

# Fail loud if the isolated env is not the treeformer-training branch.
"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

# LOWEST val_loss checkpoint from the valloss run (sort -g, not -rg).
CKPT="${CKPT:-$(ls -1 "$CKPT_DIR"/treeformer-epoch=*-val_loss=*.ckpt \
  | awk -F'val_loss=' '{print $2"\t"$0}' \
  | sort -g \
  | head -1 \
  | cut -f2-)}"
if [ -z "$CKPT" ]; then echo "ERROR: no val_loss points checkpoint in $CKPT_DIR" >&2; exit 1; fi

VIZ_DIR="outputs/viz/trained_points_896_valloss/${SPLIT}"
mkdir -p "$VIZ_DIR"
echo "=== Eval fine-tuned points (896 val_loss) on test: split=$SPLIT ==="
echo "Checkpoint: $CKPT"

"$VENV/bin/python" training/points/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-split test \
  --image-size 896 \
  --batch-size 8 \
  --num-workers 8 \
  --score-thresh 0.10 \
  --output-dir "training/points/outputs/${SPLIT}_896_valloss" \
  --viz-dir "$VIZ_DIR"

echo "=== done: training/points/outputs/${SPLIT}_896_valloss/results_${SPLIT}.txt ==="
