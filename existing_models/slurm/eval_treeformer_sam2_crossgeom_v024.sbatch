#!/bin/bash
# v0.23 leaderboard cross-geometry row: fine-tuned TreeFormer + SAM2, Run-M config.
# Follows eval_treeformer_sam2_crossgeom_v0{19,20}.sbatch, with two changes.
#
# 1. CHECKPOINT IS GLOBBED, NOT HARDCODED. The v0.19/v0.20 scripts pinned an exact
#    filename because the run already existed. This one is queued as a dependency of
#    the still-pending training job 38826334, so the epoch and val_loss are unknown
#    at submit time. Picks the LOWEST val_loss checkpoint (sort -g, ascending) from
#    the within-distribution_896_countfix run.
#
# 2. THE CHECKPOINT COMES FROM val_loss SELECTION, NOT point_recall. The v0.19/v0.20
#    cross-geom rows used checkpoints chosen by max point_recall; on v0.23 that metric
#    saturates and selected a ~1-epoch model (see ledger 38826334). Cross-geometry is
#    where that matters most: SAM2 is prompted BY the TreeFormer points, so point
#    precision propagates straight into polygon precision, and precision is exactly
#    what recall-based selection was discarding.
#
# WITHIN-DISTRIBUTION checkpoint, matching the v0.19 and v0.20 cross-geom rows so the
# version-over-version comparison stays honest. Depends on array task 0 only
# (afterok:38826334_0) rather than the whole array, so it starts as soon as the
# within-distribution split finishes instead of waiting on out-of-distribution.
#
# IMAGE SIZE: 1024, the Run-M config, unchanged from v0.19/v0.20. The checkpoint is
# trained at 896, so there is a train/eval resolution mismatch -- but a SMALLER one
# than the prior rows carried (those checkpoints were trained at 448 and evaluated at
# 1024). Holding the eval pipeline fixed keeps the delta attributable to the model.
# Set --image-size 896 instead if you want training-matched resolution, but then the
# v0.19/v0.20 comparison no longer isolates one variable.
#
# ENV: uses existing_models/treeformer_sam2/.venv via `uv run`, which is on DeepForest
# branch cursor/sam3-polygon-integration -- NOT the treeformer-training branch the
# checkpoint was trained on. Verified 2026-08-06 that a val-loss-era .ckpt loads there
# cleanly (TreeFormerModel, 730 tensors, non-trivial weights, forward OK), so this is
# not the silent-collapse case from memory points-wrong-branch-collapse. The guard
# below re-checks at run time.
#SBATCH --job-name=mt_tf_sam2_crossgeom_v024
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1

set -euo pipefail
export UV_NO_CACHE=1

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR=$REPO/existing_models/treeformer_sam2/outputs/crossgeometry_v024
CKPT_DIR="$REPO/training/points/outputs/within-distribution_896_countfix/checkpoints"

# Lowest-val_loss checkpoint from the v0.24 countfix run (sort -g, ascending).
#
# STALENESS GUARD: this dir is REUSED across dataset versions -- the v0.23 countfix
# checkpoints are still sitting in it when the v0.24 run starts, and a v0.23 checkpoint
# with a lower val_loss would silently win the sort and put a v0.23 model in a v0.24
# table row. MIN_CKPT_TIME (anything `find -newermt` accepts) restricts the glob to
# checkpoints written after that moment; the launcher sets it to its own submit time.
# Override the whole thing with: CKPT=/path/to.ckpt sbatch ...
if [ -z "${CKPT:-}" ]; then
  if [ -n "${MIN_CKPT_TIME:-}" ]; then
    echo "restricting checkpoints to those newer than: $MIN_CKPT_TIME"
    CANDIDATES=$(find "$CKPT_DIR" -maxdepth 1 -name 'treeformer-epoch=*-val_loss=*.ckpt' \
                   -newermt "$MIN_CKPT_TIME" 2>/dev/null)
  else
    echo "WARNING: MIN_CKPT_TIME unset -- stale checkpoints from a previous dataset" >&2
    echo "         version in $CKPT_DIR are eligible. Set it, or pass CKPT=." >&2
    CANDIDATES=$(find "$CKPT_DIR" -maxdepth 1 -name 'treeformer-epoch=*-val_loss=*.ckpt' 2>/dev/null)
  fi
  CKPT=$(printf '%s\n' "$CANDIDATES" | grep . \
    | awk -F'val_loss=' '{print $2"\t"$0}' \
    | sort -g \
    | head -1 \
    | cut -f2-)
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "ERROR: no val_loss checkpoint in $CKPT_DIR" >&2
  echo "       (training job 38826334_0 must have completed and written one)" >&2
  exit 1
fi

cd $REPO/existing_models/treeformer_sam2

echo "=== v0.23 within-FT TreeFormer + SAM2 cross-geometry (Run-M config) ==="
echo "checkpoint: $CKPT"

# Fail loud if the checkpoint does not restore a working TreeFormerModel in this env.
# A silently-wrong restore is the documented collapse mode for point models loaded
# against a mismatched DeepForest branch, and it costs a full 24h eval to discover.
./.venv/bin/python -c "
import sys, warnings; warnings.filterwarnings('ignore')
from deepforest import main as df_main
m = df_main.deepforest.load_from_checkpoint('$CKPT', weights_only=False,
        config_args={'devices':1,'accelerator':'cpu','workers':0})
cls = type(m.model).__name__
sd = m.model.state_dict()
nz = sum(int((v!=0).sum()) for v in sd.values() if v.dtype.is_floating_point)
print(f'ckpt guard: {cls}, {len(sd)} tensors, {nz} nonzero elems')
assert cls == 'TreeFormerModel', f'expected TreeFormerModel, got {cls}'
assert nz > 1_000_000, f'weights look degenerate: only {nz} nonzero'
"

HF_TOKEN_FILE=$REPO/.hf_token
if [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi

uv run python eval_polygons_crossgeometry.py \
  --root-dir "$ROOT_DIR" \
  --split-scheme crossgeometry \
  --checkpoint "$CKPT" \
  --device cuda \
  --batch-size 16 \
  --num-workers 8 \
  --negative-prompts \
  --sam2-model "facebook/sam2.1-hiera-large" \
  --prompt-batch-size 256 \
  --score-thresh-tf 0.3 \
  --sam2-accept-score 0.1 \
  --mask-threshold 0.3 \
  --image-size 1024 \
  --output-dir "$OUT_DIR" \
  --viz-dir "$OUT_DIR/viz" \
  --viz-size 1024

echo "=== Done. Results in $OUT_DIR ==="
