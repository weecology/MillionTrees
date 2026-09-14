#!/bin/bash
#SBATCH --job-name=mt_d2_ablation_smoke
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Smoke test for the NATIVE DETECTRON2 polygon weak-supervision ablation. Tiny caps
# everywhere; this exists to prove the plumbing before the real run burns GPU-days.
#
# The DeepForest/torchvision box round already answered "does weak-box pretraining help"
# with a null (co-training won within-distribution by +0.019 AP40 and lost OOD). This
# rebuilds the same three arms on Detectron2 -- the stack that actually produces the
# leaderboard polygon numbers -- so the null can be pinned on the weak labels rather than
# on DeepForest.
#
# Three things have to hold before the real submit:
#
#   1. STAGE 1 trains a native D2 Mask R-CNN with the mask branch off on the weak boxes,
#      verifies COCO init tensor-for-tensor (281 backbone/FPN), selects its checkpoint by
#      holdout box_recall rather than loss, and exports 295 transferable tensors.
#   2. ARM 1 (sequential) merges that export onto COCO and refuses to run if the backbone
#      came out byte-identical to COCO -- i.e. a control in disguise.
#   3. ARM 2 (co-training) registers the weak box-only tiles alongside the polygon tiles
#      and trains without the mask head ever seeing a rectangle.
#
# NOTE ON SIZE: this smoke does NOT pass --mini. TreePolygonsDataset ignores mini=True
# whenever include_unsupervised is False (the default): the "select supervised-only"
# branch overwrites _dataset_name with 'TreePolygons_supervised' after the mini versions
# dict has been chosen, so the loader lands on the full release either way. The first
# attempt (40200530) timed out at 1h because of it -- the 10-iteration training step was
# instant and the *evaluation* over 1,742 full-release test tiles ate the wall clock.
# Smallness comes from --max-iter plus --eval-split validation (59 tiles) instead.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SMOKE_DIR="$REPO/training/weak_supervision/outputs/d2_ablation_smoke"
rm -rf "$SMOKE_DIR"; mkdir -p "$SMOKE_DIR"

# Detectron2 (CanopyRS venv) CUDA ops are a fat binary for L4 (sm_89) + B200 (sm_100).
module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Block-buffered stdout makes a running job unreadable; the ledger workflow reads .out
# while the job is still going.
export PYTHONUNBUFFERED=1

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

echo "############ 1/3  STAGE 1: native D2 weak-box pretraining ############"
python "$REPO/training/polygons/pretrain_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme within-distribution \
  --limit-weak-images 32 \
  --holdout-images 16 \
  --batch-size 2 \
  --max-iter 20 \
  --eval-period 10 \
  --lr 1e-3 \
  --num-workers 8 \
  --output-dir "$SMOKE_DIR/stage1"

STAGE1_EXPORT="$SMOKE_DIR/stage1/d2_network_within-distribution.pkl"
test -f "$STAGE1_EXPORT" || { echo "ERROR: stage-1 export missing" >&2; exit 1; }

# Co-training first: it is the arm whose plumbing is still unproven. Arm 1 already
# produced its receipts in 40200530 ("Stage-1 init confirmed: 58 backbone/FPN tensors
# differ from COCO", "Loaded FULL pretrained network: 295 stage-1 tensors").
echo
echo "############ 2/3  ARM 2: co-training (weak boxes + polygons) ############"
python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme within-distribution \
  --init-mode cotrain \
  --limit-weak-images 16 \
  --batch-size 2 \
  --max-iter 10 \
  --lr 0.01 \
  --num-workers 8 \
  --eval-split validation \
  --output-dir "$SMOKE_DIR/arm2_cotrain"

echo
echo "############ 3/3  ARM 1: sequential (stage-1 whole network) ############"
python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme within-distribution \
  --init-mode box_pretrained_full \
  --weights "$STAGE1_EXPORT" \
  --batch-size 2 \
  --max-iter 10 \
  --lr 0.01 \
  --num-workers 8 \
  --eval-split validation \
  --output-dir "$SMOKE_DIR/arm1_pretrained"

echo
echo "SMOKE PASSED"
