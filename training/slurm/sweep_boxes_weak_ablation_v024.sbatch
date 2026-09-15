#!/bin/bash
# Score-threshold sweep for the v0.24 box->box weak-supervision ablation arms.
#
# Question: is over-prediction (many low-confidence boxes) depressing the reported
# AP40/F1 of the box-pretrained arm relative to the COCO control? If F1 peaks well
# above the 0.1 operating point for the box-pretrained arm but not for COCO, a
# per-model calibrated threshold would close part of the gap and the "weak
# pretraining is null" read is partly an operating-point artifact. If both arms
# peak near 0.1, the tie is real.
#
# Eval-only, no retraining. Inference runs once per arm with the model emission
# threshold driven to 0.01; run_threshold_sweep re-scores recall / mask-aware
# precision / AP40 / F1 across the default grid (0.05..0.9). Full v0.24 test set.
#
# Array: 0-3 = {within-distribution, out-of-distribution} x {box_pretrained_full, coco}
#SBATCH --job-name=mt_sweep_weak_box_v024
#SBATCH --output=/home/b.weinstein/logs/%x_%A_%a.out
#SBATCH --error=/home/b.weinstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=90GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=/blue/ewhite/b.weinstein/src/MillionTrees:$PYTHONPATH

cd /blue/ewhite/b.weinstein/src/MillionTrees

ROOT_DIR=/orange/ewhite/web/public/MillionTrees
ABLATION_DIR=training/boxes/outputs/converged_ablation_v024

case "$SLURM_ARRAY_TASK_ID" in
  0) SPLIT=within-distribution ; ARM=box_pretrained_full_converged ; SUB=""                    ;;
  1) SPLIT=within-distribution ; ARM=coco                          ; SUB=""                    ;;
  2) SPLIT=out-of-distribution ; ARM=box_pretrained_full_converged ; SUB="out-of-distribution/" ;;
  3) SPLIT=out-of-distribution ; ARM=coco                          ; SUB="out-of-distribution/" ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2 ; exit 1 ;;
esac

CKPT="$ABLATION_DIR/${SUB}${ARM}/checkpoints/boxes-best.ckpt"
OUT_DIR="$ABLATION_DIR/${SUB}${ARM}/sweep"
echo "Split: $SPLIT   Arm: $ARM"
echo "Checkpoint: $CKPT"
test -f "$CKPT" || { echo "ERROR: checkpoint not found" >&2; exit 1; }

srun uv run python training/boxes/eval.py \
  --checkpoint "$CKPT" \
  --root-dir "$ROOT_DIR" \
  --split-scheme "$SPLIT" \
  --batch-size 8 \
  --num-workers 8 \
  --score-threshold 0.01 \
  --sweep \
  --viz-dir "" \
  --output-dir "$OUT_DIR"
