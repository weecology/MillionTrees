#!/bin/bash
#SBATCH --job-name=mt_pretrain_viz_smoke
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Smoke test for the Comet detection overlays added to stage-1 pretraining.
# The supervised-validation instrumentation itself is already verified (39879655);
# this run only checks the new visualisation path, in ~15 min instead of ~7 h:
#   1. "[viz] Comet detection overlays enabled for panels ['train-weak',
#      'val-supervised']" appears -- both panels built, neither empty
#   2. images land in the Comet Graphics tab at step 0 (COCO init) and at
#      steps 1-2 (after each epoch), under stable names train-weak/NN and
#      val-supervised/NN so the epoch slider works
#   3. each image carries TWO toggleable annotation layers, "Ground truth" and
#      "Predictions", and the predictions carry scores
#   4. viz_*_mean_dets_per_img / _mean_gt_per_img / _mean_max_score are logged
#      as curves -- the scalar companions that say spraying vs. silent
#   5. the run still completes and exports weights, i.e. the callback cannot
#      break training (empty-prediction tiles included, which is the normal
#      early-epoch state for this stage)
#   6. --ckpt-monitor box_recall (mode=max) RESOLVES: the checkpoint section must
#      print a non-empty best path and a numeric score, not a Lightning "metric
#      not found in the returned metrics" warning. box_recall is computed in
#      on_validation_epoch_end and only when val_accuracy_interval allows, so it
#      is not obviously available to ModelCheckpoint -- and every previous step of
#      this script hid exactly one silent gate of that shape. Worth 4 minutes
#      before committing a 7 h run to it. (The score itself will be 0.0 at this
#      budget; that is expected and not what is being tested.)
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

OUT_DIR="training/weak_supervision/outputs/pretrain_viz_smoke"
mkdir -p "$OUT_DIR"

uv run python training/boxes/pretrain_backbone_for_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme within-distribution \
  --include-unsupervised \
  --init-mode coco \
  --batch-size 8 \
  --max-epochs 2 \
  --lr 1e-4 \
  --scheduler cosine \
  --supervised-val-images 64 \
  --num-workers 4 \
  --limit-train-batches 30 \
  --limit-val-batches 8 \
  --eval-max-batches 8 \
  --val-every-n-epochs 1 \
  --ckpt-monitor box_recall \
  --ckpt-mode max \
  --comet \
  --comet-name boxes-within-distribution-pretrain-supval-vizsmoke \
  --viz-images 4 \
  --output-dir "$OUT_DIR"
