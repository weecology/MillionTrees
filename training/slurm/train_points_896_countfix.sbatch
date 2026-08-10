#!/bin/bash
# TreeFormer point retrain at image_size=896 with the COUNT LOSS ACTUALLY ENABLED.
#
# WHY THIS EXISTS: every MillionTrees point fine-tune to date (38750849, 38826334,
# 38936192, 38936193) inherited DeepForest's default enforce_count=True from
# conf/point.yaml. With enforce_count on and a GT count available on the training
# path, _normalize_density rescales the density map so its sum EXACTLY equals the
# ground-truth count, so
#     count_loss = |log1p(pred_sum) - log1p(gt_count)| = |x - x| = 0
# identically, with a zero gradient. Verified standalone: density_map sums forced
# from ~5e5 to exactly [37, 91] for a batch with 37/91 points, count_loss 0.0,
# count_mae 0.0, d(count_loss)/d(score_map) 0.0. Confirmed in the runs themselves:
# count_loss and count_mae logged 0.00000 at EVERY epoch, and val_loss equalled
# val_density_l1_loss to 4 dp at every epoch on both splits.
#
# Consequence: the model learned WHERE trees are (OT + density L1) but never HOW
# MANY. At inference enforce_count falls back to the CLS count head, which got no
# usable gradient either, so the density is scaled by a garbage count and peak
# extraction over-fires ~5x. Fine-tuning therefore DESTROYED the count calibration
# the released checkpoint already had: out-of-distribution macro counting nMAE
# 0.489 -> 0.901, slope 0.158 -> -0.922, held-out Allen MAE 46.9 -> 302.2, while
# precision fell 0.737 -> 0.538 for a recall gain of 0.763 -> 0.850 (F1 0.750 ->
# 0.659).
#
# FIX: --loss-preset pretrain, which mirrors conf/point_pretrain.yaml -- the config
# the released weecology checkpoint was actually trained under:
#   enforce_count: false          <-- the count loss becomes live
#   losses: [count, ot, density_l1]
#   mae_weight: 0.025             (not the dataclass default of 1.0)
#   density_l1_weight: 0.05
#   score_integration_radius: 2   (matches what eval uses)
#
# Checkpoint selection stays on val_loss with early stopping off, matching 38826334
# so the only deliberate change vs that run is the loss configuration. NOTE val_loss
# is NOT comparable across the two: with the count term live it is a different
# quantity. Compare on the eval metrics (recall / mask-aware precision / F1 /
# counting nMAE and slope), not on the loss curve.
#
# Runs in the frozen .venv-treeformer (DeepForest treeformer-training branch),
# called directly -- NOT `uv run`: the shared .venv is pinned to the polygon
# Mask R-CNN branch, which silently collapses TreeFormer point training
# (memory points-wrong-branch-collapse, ledger 36162735).
#SBATCH --job-name=mt_train_points_896_countfix
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
  --loss-preset pretrain \
  --ckpt-monitor val_loss \
  --no-early-stop \
  --output-dir "training/points/outputs/${SPLIT}_896_countfix" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-896-countfix" \
  --comet
