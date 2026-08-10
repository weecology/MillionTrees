#!/bin/bash
#SBATCH --job-name=mt_eval_val_points
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Manuscript-final validation: TreeFormer point model on the held-out Allen et al. 2025
# + Frey et al. 2026 TLS set. Uses the 896 image-size checkpoint (point_recall ~0.89 OOD)
# from the isolated-venv run (per user: most up-to-date hyperparameters). NOTE the points
# validation split holds BOTH sources -- read them from the per-source breakdown.
#
# Runs in the frozen .venv-treeformer (DeepForest treeformer-training branch), called
# directly -- NOT `uv run`. The shared .venv is pinned to the polygon Mask R-CNN branch,
# which silently collapses TreeFormer point eval/training (see ledger 36162735, memory
# points-wrong-branch-collapse).
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

VENV=/blue/ewhite/b.weinstein/src/MillionTrees/.venv-treeformer
ROOT_DIR=/orange/ewhite/web/public/MillionTrees
SPLIT=out-of-distribution

# Which training run to score. Defaults to the point_recall-selected _896_isolated
# run; override for a different selection, e.g. the val_loss-selected run:
#   OUT_DIR=training/points/outputs/out-of-distribution_896_valloss \
#   CKPT=.../treeformer-epoch=19-val_loss=0.4902.ckpt \
#   sbatch eval_validation_points.sbatch
# OUT_DIR must be set alongside CKPT, or the results land in the wrong run's dir.
OUT_DIR="${OUT_DIR:-training/points/outputs/${SPLIT}_896_isolated}"
CKPT_DIR="$OUT_DIR/checkpoints"

# Fail loud if the isolated env is not the treeformer-training branch.
"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

# Highest point_recall checkpoint from the 896 isolated run.
# Override at run time with: CKPT=/path/to.ckpt sbatch eval_validation_points.sbatch
CKPT="${CKPT:-$(ls -1 "$CKPT_DIR"/treeformer-epoch=*-point_recall=*.ckpt \
  | awk -F'point_recall=' '{print $2"\t"$0}' \
  | sort -rg \
  | head -1 \
  | cut -f2-)}"
if [ -z "$CKPT" ]; then echo "ERROR: no 896 isolated points checkpoint in $CKPT_DIR" >&2; exit 1; fi

VIZ_DIR="outputs/viz/validation_points"
mkdir -p "$VIZ_DIR"
echo "=== Eval points (896 iso) on validation (Allen+Frey): split=$SPLIT ==="
echo "Checkpoint: $CKPT"

# image-size 896 matches the checkpoint; score-thresh 0.10 + default eval_score_threshold
# (0.4) + default score_integration_radius (2) match the leaderboard points eval.
# Results -> results_<split>_validation.{txt,json} in the 896 dir (test outputs untouched).
"$VENV/bin/python" training/points/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --eval-split validation \
  --image-size 896 \
  --batch-size 8 \
  --num-workers 8 \
  --score-thresh 0.10 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$VIZ_DIR"

echo "=== done: $OUT_DIR/results_${SPLIT}_validation.txt ==="
