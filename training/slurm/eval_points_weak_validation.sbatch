#!/bin/bash
# Score both arms of the TreePoints weak-supervision ablation on the MillionTrees
# points VALIDATION split (Allen et al. 2025 + Frey et al. 2026 TLS). train.py
# already scored TEST at the end of each stage-2 run; this fills the val column.
#
# Array index -> (arm, split):
#   0 control  within-distribution   2 treatment within-distribution
#   1 control  out-of-distribution   3 treatment out-of-distribution
#
# Submit with an afterok dependency on the stage-2 jobs. Runs in .venv-treeformer.
#SBATCH --job-name=mt_points_weak_valeval
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

VENV=/blue/ewhite/b.weinstein/src/MillionTrees/.venv-treeformer
ROOT_DIR=/orange/ewhite/web/public/MillionTrees

ARMS=(control control treatment treatment)
SPLITS=(within-distribution out-of-distribution within-distribution out-of-distribution)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}
OUT_DIR="training/points/weak_supervision_outputs/${ARM}/${SPLIT}"
CKPT_DIR="$OUT_DIR/checkpoints"

"$VENV/bin/python" -c "import json,glob; d=json.load(open(glob.glob('$VENV/lib/python*/site-packages/deepforest-*/direct_url.json')[0])); assert d['vcs_info']['requested_revision']=='treeformer-training', d; print('deepforest branch OK:', d['vcs_info']['requested_revision'])"

# Lowest val_loss checkpoint from the stage-2 run.
CKPT=$(ls -1 "$CKPT_DIR"/treeformer-epoch=*-val_loss=*.ckpt \
  | awk -F'val_loss=' '{print $2"\t"$0}' | sort -g | head -1 | cut -f2-)
if [ -z "$CKPT" ]; then echo "ERROR: no val_loss checkpoint in $CKPT_DIR" >&2; exit 1; fi
echo "=== eval $ARM / $SPLIT on validation: $CKPT ==="

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
  --viz-dir "$OUT_DIR/viz_validation"

echo "=== done: $OUT_DIR/results_${SPLIT}_validation.txt ==="
