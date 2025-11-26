#!/bin/bash
#SBATCH --job-name=mt-examples
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60GB

# Common args
ROOT_DIR=${MT_ROOT:-/orange/ewhite/web/public/MillionTrees}
OUT_DIR=${OUT_DIR:-runs/examples/${SLURM_JOB_ID:-local}}
mkdir -p "$OUT_DIR"

## Random split
uv run python scripts/run_examples.py --root-dir $ROOT_DIR --split-scheme random --mini --device cuda --max-batches 2
uv run python scripts/run_examples.py --root-dir $ROOT_DIR --split-scheme random --device cuda

## Zero-shot split
uv run python scripts/run_examples.py --root-dir $ROOT_DIR --split-scheme zeroshot --mini --device cuda --max-batches 2
uv run python scripts/run_examples.py --root-dir $ROOT_DIR --split-scheme zeroshot --device cuda

## Cross-geometry split (not ready)
# uv run python scripts/run_examples.py --root-dir $ROOT_DIR --split-scheme crossgeometry --mini --device cuda --max-batches 2