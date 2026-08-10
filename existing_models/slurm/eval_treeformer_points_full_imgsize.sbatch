#!/bin/bash
# Confirm at full-release scale the SmallTreePoints finding (job 34694394):
# eval image_size=448 under-resolves dense canopy; image_size=896 ~doubles point
# recall (0.27->0.55 on small) at high precision and halves counting MAE. Run the
# pretrained TreeFormer baseline at 448 (control) and 896 on the FULL release,
# random + zeroshot. Inference only (1014 test imgs). Produces the corrected
# leaderboard point numbers. score_thresh=0.1, integration_radius=2 (standard),
# eval_score_threshold left at the dataset default (0.4 had ~no effect, E0/E1).
#SBATCH --job-name=mt_tf_points_full_imgsize
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1

set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees/existing_models/treeformer
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:${PYTHONPATH:-}

ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
SPLITS=(random zeroshot)
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

for IMG in 448 896; do
  OUT_DIR="outputs/full_${SPLIT}_img${IMG}"
  echo "######## split=${SPLIT} image_size=${IMG} ########"
  uv run python eval_points.py \
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

echo "=== Done. Results in outputs/full_${SPLIT}_img*/ ==="
