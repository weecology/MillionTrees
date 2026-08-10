#!/bin/bash
#SBATCH --job-name=mt_poly_d2_evalonly
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1

# Validate the (edited, 0.15-gate) Detectron2 eval/finalize/write path on the
# smoke run's saved model_final.pth -- no retraining. The smoke training was
# proven; its 1h job timed out mid-eval at the old 0.05 gate. This confirms
# detectron2_predict_for_eval + TreePolygonsStreamingEvalState.finalize + the
# txt/json writes complete and emit sane MillionTrees metrics before we rely on
# the full array's eval.
set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}

module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=/apps/compilers/cuda/12.8.1
export PATH="$CUDA_HOME/bin:$PATH"

cd "$REPO/existing_models/canopyrs"
source .venv/bin/activate

python "$REPO/training/polygons/train_detectron2.py" \
  --root-dir "$ROOT_DIR" \
  --split-scheme within-distribution \
  --eval-only \
  --batch-size 8 \
  --num-workers 8 \
  --output-dir "$REPO/training/polygons/outputs/detectron2_smoke"

echo "=== Detectron2 polygon eval-only validation done ==="
