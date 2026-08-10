#!/bin/bash
# Matched-resolution retrain of the TreeFormer point model at image_size=896 for
# both splits, RE-RUN with an ISOLATED, frozen DeepForest environment.
#
# WHY THIS EXISTS: the overnight 896 run (job 36103736) collapsed to keypoint_acc
# = 0.000 on both splits. Root cause was NOT the image size — the shared .venv had
# been left pinned (via [tool.uv.sources] + `uv sync`) to the POLYGON Mask R-CNN
# branch (bw4sz/DeepForest@cursor/polygon-maskrcnn-jv-with-safecrop) for the Table 6
# polygon work. That branch happens to ship a treeformer.py, so the point script
# loaded a TreeFormerModel and ran to completion, but its point-training path is
# broken → collapse. The last GOOD point run (35991396, keypoint_acc 0.542) used
# jveitchmichaelis/DeepForest@treeformer-training.
#
# FIX: this job runs against a dedicated, frozen venv built once from the
# treeformer-training branch (.venv-treeformer, deepforest commit aaad58a). It
# calls that venv's python DIRECTLY instead of `uv run --group treeformer`, so a
# future polygon `uv sync` on the shared .venv can never change the DeepForest
# branch out from under a point run. Rebuild instructions:
#   (temporarily flip [tool.uv.sources] deepforest -> jveitchmichaelis@treeformer-training)
#   UV_PROJECT_ENVIRONMENT=.venv-treeformer uv sync --group treeformer --no-group polygon --no-group dev
#   (then revert the pin)
#
# Recipe mirrors the standard leaderboard point run (pretrained init, lr 2e-4, 20
# epochs); only resolution (896 vs 448) differs vs the 448 leaderboard rows.
# Output to *_896_isolated dirs — a separate experiment, NOT the leaderboard dir.
#SBATCH --job-name=mt_train_points_896_iso
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
  --output-dir "training/points/outputs/${SPLIT}_896_isolated" \
  --comet-name "points-${SPLIT}-pretrained-lr2e-4-896-isoenv" \
  --comet
