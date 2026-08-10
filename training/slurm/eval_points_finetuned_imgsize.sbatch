#!/bin/bash
# E7 (docs/treeformer_points_diagnosis_plan.md): re-evaluate the FINE-TUNED
# TreeFormer checkpoints at the chosen leaderboard resolution. The "fine-tuning
# does nothing" claim was made at image_size=448, where both pretrained and
# fine-tuned models were resolution-starved. Re-eval the best (point_recall)
# checkpoint per split at 448 (sanity vs current table) and 672 (comparable to
# the re-scored pretrained rows) to see if fine-tuning shows value once the
# resolution bottleneck is removed. NOTE: these checkpoints were TRAINED at 448,
# so 672 is a train/eval resolution mismatch — interpret accordingly; a clean
# test needs retraining at 672 (follow-up).
#SBATCH --job-name=mt_tf_points_ft_imgsize
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

if [ "$SPLIT" = "random" ]; then
  CKPT="training/points/outputs/random/checkpoints/treeformer-epoch=19-point_recall=0.2408.ckpt"
else
  CKPT="training/points/outputs/zeroshot/checkpoints/treeformer-epoch=00-point_recall=0.1286.ckpt"
fi

for IMG in 448 672; do
  OUT_DIR="training/points/outputs/${SPLIT}/eval_ft_img${IMG}"
  echo "######## FINE-TUNED split=${SPLIT} image_size=${IMG} ckpt=${CKPT} ########"
  uv run python -m training.points.eval \
    --checkpoint "$CKPT" \
    --root-dir "$ROOT_DIR" \
    --split-scheme "$SPLIT" \
    --score-thresh 0.1 \
    --score-integration-radius 2 \
    --image-size "$IMG" \
    --batch-size 8 \
    --num-workers 8 \
    --viz-dir "" \
    --output-dir "$OUT_DIR"
done

echo "=== Done. Results in training/points/outputs/${SPLIT}/eval_ft_img*/ ==="
