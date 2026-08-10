#!/bin/bash
#SBATCH --job-name=mt_eval_points_896_iso
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Test-split eval of the fine-tuned TreeFormer point model at image_size=896 (the
# checkpoints from train_points_896_isolated.sbatch, i.e. the config the validation
# table already uses). Re-run so the points rows of the leaderboard sit on the same
# dataset version (v0.22) as every other row of the AP40 refresh. Points have no AP
# metric; this only re-bases keypoint accuracy / precision / counting MAE.
#
# Runs in the frozen .venv-treeformer (DeepForest treeformer-training branch), called
# directly -- NOT `uv run`: the shared .venv is pinned to the polygon Mask R-CNN
# branch, which silently collapses TreeFormer point eval (memory
# points-wrong-branch-collapse, ledger 36162735).
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

VENV=/blue/ewhite/b.weinstein/src/MillionTrees/.venv-treeformer
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
CKPT_DIR="training/points/outputs/${SPLIT}_896_isolated/checkpoints"

# Fail loud if the isolated env is not the treeformer-training branch.
"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

# Highest point_recall checkpoint from the 896 isolated run.
CKPT=$(ls -1 "$CKPT_DIR"/treeformer-epoch=*-point_recall=*.ckpt \
  | awk -F'point_recall=' '{print $2"\t"$0}' \
  | sort -rg \
  | head -1 \
  | cut -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no 896 isolated points checkpoint in $CKPT_DIR" >&2; exit 1; fi

VIZ_DIR="outputs/viz/trained_points_896_iso/${SPLIT}"
mkdir -p "$VIZ_DIR"
echo "=== Eval fine-tuned points (896 iso) on test: split=$SPLIT ==="
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
  --output-dir "training/points/outputs/${SPLIT}_896_isolated" \
  --viz-dir "$VIZ_DIR"

echo "=== done: training/points/outputs/${SPLIT}_896_isolated/results_${SPLIT}.txt ==="
