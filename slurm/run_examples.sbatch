#!/bin/bash
# Quick sanity-check job: runs existing-model evals on mini datasets.
# For the full benchmark pipeline use slurm/run_benchmark.sbatch instead.
#
# Usage: sbatch slurm/run_examples.sbatch
#SBATCH --job-name=mt-examples
#SBATCH --output=/home/b.weinstein/logs/%x_%j.out
#SBATCH --error=/home/b.weinstein/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB

set -euo pipefail

REPO=/blue/ewhite/b.weinstein/src/MillionTrees
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
cd "$REPO"

for SPLIT in random zeroshot; do
  echo "=== DeepForest mini eval: split=$SPLIT ==="
  cd "$REPO/existing_models/deepforest"
  uv run python eval_boxes.py   --root-dir "$ROOT_DIR" --split-scheme "$SPLIT" --mini --max-batches 2
  uv run python eval_points.py  --root-dir "$ROOT_DIR" --split-scheme "$SPLIT" --mini --max-batches 2
  uv run python eval_polygons.py --root-dir "$ROOT_DIR" --split-scheme "$SPLIT" --mini --max-batches 2

  echo "=== SAM3 mini eval: split=$SPLIT ==="
  cd "$REPO/existing_models/sam3"
  uv run python eval_boxes.py   --root-dir "$ROOT_DIR" --split-scheme "$SPLIT" --device cuda --mini --max-batches 2
  uv run python eval_points.py  --root-dir "$ROOT_DIR" --split-scheme "$SPLIT" --device cuda --mini --max-batches 2
  uv run python eval_polygons.py --root-dir "$ROOT_DIR" --split-scheme "$SPLIT" --device cuda --mini --max-batches 2
done
