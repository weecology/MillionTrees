#!/bin/bash
#SBATCH --job-name=mt_pretrain_converged
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# Stage-1 pretraining, best recipe the LR sweep (39879762) found, with the two
# defects that sweep exposed now fixed.
#
# WHAT THE SWEEP SETTLED. Its decision rule was: if val_loss falls substantially
# further at higher LR than under the old recipe, stage 1 was undertrained and the
# whole weak-supervision null is an artifact. It did not. Best supervised val_loss
# came out a tie -- oldrecipe (b8, lr1e-4 constant) 1.4297 vs lr1e-3-cosine (b32)
# 1.4218, a 0.008 gap -- so the null is NOT simply a training-budget artifact and
# the pretraining signal itself is the remaining suspect. This run is therefore not
# a re-litigation of that question; it is the clean, converged stage-1 network the
# next round of that investigation needs, produced without the two known defects:
#
#   1. NO GRADIENT CLIPPING. lr1e-2 (task 2) went NaN from epoch 0 and was killed,
#      and lr1e-3 (task 1) diverged late -- val_loss 1.42 at epoch 1 climbing to
#      6.45, val_classification to 5.73. Stage 2 has always run at lr 1e-2 without
#      trouble precisely because train.py passes --clip 1.0; stage 1 had none.
#      --grad-clip 1.0 makes this the same recipe stage 2 already survives.
#
#   2. val_loss WAS THE WRONG SELECTION METRIC. Both sweep arms exported their
#      val_loss-best epoch, and for lr1e-3 that was epoch 1: recall 0.003,
#      mask-aware precision 0.116, AP40 0.001 -- a near-dead detector that
#      nonetheless won on loss. oldrecipe's checkpoint had WORSE val_loss and was
#      10x better at detecting (recall 0.041 / 0.258 / AP40 0.006). Meanwhile the
#      per-epoch box_recall each arm actually reached -- 0.237 and 0.176 -- was
#      thrown away. Selecting on box_recall (mode=max) exports the epoch that
#      detects trees rather than the epoch that minimises a loss dominated by the
#      classification term on 143-boxes-per-tile pseudo-labels.
#
# Only the recipe defects change. Epoch budget stays at 20 with cosine T_max=20, and
# the split stays within-distribution, so this is directly comparable to sweep task 1
# rather than confounded with a longer schedule. --num-workers 16 (up from 4) is pure
# speed: GPU utilisation was 13-22% at 4 workers, i.e. stage 1 is I/O bound.
#
# --viz-images 8 turns on the Comet detection overlays (training/boxes/comet_viz.py):
# the same 8 unsupervised tiles and 8 held-out supervised tiles logged every epoch
# with ground-truth and prediction layers, because the scalar curves cannot
# distinguish "learning slowly" from "fitting noise" on 7.1M weak boxes.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

# SPLIT is overridable so the identical recipe can produce the out-of-distribution
# stage-1 network (job 39974598 was the within-distribution one). Default is unchanged,
# so the WD invocation above is byte-identical to what produced
# box_network_within-distribution.pt. Out-of-distribution writes to its own subdirectory
# because the exports are split-named but checkpoints/ is not.
SPLIT="${SPLIT:-within-distribution}"
TAG=lr1e-3-cosine-clip
OUT_DIR="training/weak_supervision/outputs/pretrain_converged/$TAG"
if [ "$SPLIT" != "within-distribution" ]; then
  OUT_DIR="$OUT_DIR/$SPLIT"
fi
mkdir -p "$OUT_DIR"

uv run python training/boxes/pretrain_backbone_for_polygons.py \
  --root-dir /orange/ewhite/web/public/MillionTrees \
  --split-scheme "$SPLIT" \
  --include-unsupervised \
  --init-mode coco \
  --batch-size 32 \
  --max-epochs 20 \
  --lr 1e-3 \
  --scheduler cosine \
  --grad-clip 1.0 \
  --supervised-val-images 512 \
  --val-every-n-epochs 1 \
  --ckpt-monitor box_recall \
  --ckpt-mode max \
  --viz-images 8 \
  --num-workers 16 \
  --output-dir "$OUT_DIR" \
  --comet-name "boxes-${SPLIT}-pretrain-supval-${TAG}" \
  --comet
