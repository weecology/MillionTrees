#!/bin/bash
#SBATCH --job-name=mt_pretrain_converged_v024
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

# v0.24 re-run of the converged stage-1 box pretraining (was job 39974598 WD on
# v0.23). Recipe byte-identical to training/slurm/pretrain_backbone_converged.sbatch:
# --grad-clip 1.0, --ckpt-monitor box_recall --ckpt-mode max, lr 1e-3 cosine, 20
# epochs, 512-image supervised holdout. ONLY the dataset version moves (the loaders
# default to the latest _versions_dict key, now 0.24) and the output directory, which
# is suffixed _v024 so the v0.23 converged checkpoints/exports are not clobbered.
#
# Why re-run for v0.24: the box->box ablation's entire interpretation rides on the
# out-of-distribution row, and v0.24 repairs the OOD split (SelvaBox -> train-only,
# NEON MultiTemporal added to the box hold-out, leaked train rows removed). The
# v0.23 converged numbers are not citable against the repaired split.
#
# v0.24 note: stage 1 selects include_sources=['*unsupervised*']. If AutoArborist
# (Beery et al. 2022, relabelled unsupervised in v0.24) has box rows it now enters
# the weak pretraining set; the co-training arm picks up the same source via
# --include-unsupervised, so the two weak routes stay matched.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

SPLIT="${SPLIT:-within-distribution}"
TAG=lr1e-3-cosine-clip
OUT_DIR="training/weak_supervision/outputs/pretrain_converged_v024/$TAG"
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
  --comet-name "boxes-${SPLIT}-pretrain-supval-${TAG}-v024" \
  --comet
