#!/bin/bash
# Pretrained TreeFormer point baseline at image_size=896, matched to the
# fine-tuned 896 run (training/slurm/train_points_896_isolated.sbatch,
# job 36162735) so the leaderboard can adopt 896 as the standard resolution
# with an apples-to-apples pretrained comparison.
#
# Uses the ISOLATED, frozen .venv-treeformer (deepforest@treeformer-training,
# commit aaad58a) instead of `uv run`, because the shared .venv is currently
# pinned (pyproject.toml [tool.uv.sources]) to the polygon Mask R-CNN branch
# for the Table 6 work — running this off the shared venv would silently load
# the wrong treeformer.py (see train_points_896_isolated.sbatch for the same
# failure mode on the fine-tuned side).
#SBATCH --job-name=mt_eval_treeformer_896_iso
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
VENV=$REPO/.venv-treeformer
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(within-distribution out-of-distribution)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_BASE=$REPO/existing_models/treeformer/outputs/${SPLIT}_896_isolated

cd $REPO/existing_models/treeformer

# Fail loud if the isolated env is not the treeformer-training branch.
"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

echo "=== TreeFormer pretrained baseline (no MT training), image_size=896: split=$SPLIT ==="

"$VENV/bin/python" eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --image-size 896 \
  --batch-size 8 \
  --num-workers 4 \
  --score-thresh 0.10 \
  --output-dir "$OUT_BASE"

echo "=== TreeFormer eval done. Results in $OUT_BASE ==="
