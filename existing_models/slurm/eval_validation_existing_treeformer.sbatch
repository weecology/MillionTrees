#!/bin/bash
#SBATCH --job-name=mt_eval_val_treeformer
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

# Pretrained TreeFormer point baseline (no MT training, zero-shot) on the held-out TLS
# validation set (Allen et al. 2025 + Frey et al. 2026), at image_size=896 to match the
# fine-tuned points validation run (896 isolated ckpt). Zero-shot => split-independent.
#
# Runs in the frozen .venv-treeformer (deepforest@treeformer-training) called directly --
# NOT `uv run` -- because the shared .venv is pinned to the polygon branch and collapses
# TreeFormer (see eval_treeformer_896_isolated.sbatch / memory points-wrong-branch-collapse).
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
VENV=$REPO/.venv-treeformer
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_BASE=$REPO/existing_models/treeformer/outputs/validation_896_isolated

cd $REPO/existing_models/treeformer

# Fail loud if the isolated env is not the treeformer-training branch.
"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

echo "=== TreeFormer pretrained eval on validation (Allen+Frey), image_size=896 ==="

"$VENV/bin/python" eval_points.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme out-of-distribution \
  --eval-split validation \
  --image-size 896 \
  --batch-size 8 \
  --num-workers 4 \
  --score-thresh 0.10 \
  --output-dir "$OUT_BASE"

echo "=== done. Results in $OUT_BASE ==="
