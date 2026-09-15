#!/usr/bin/env bash
# TreePoints weak-supervision ablation -- point analogue of the box/polygon arms
# in this directory. Two-stage, per split:
#
#   stage 1  pretrain_points_autoarborist.sbatch  (array 0-1: WD + OOD)
#              continue-pretrain weecology/deepforest-tree-point on AutoArborist
#              -> training/points/weak_supervision_outputs/pretrain/<split>/hf_checkpoint
#
#   stage 2  control    train_points_weak_control.sbatch    (array 0-1, no dep)
#              fine-tune on supervised points train FROM the released checkpoint
#            treatment  train_points_weak_treatment.sbatch  (per split, afterok stage 1)
#              fine-tune on supervised points train FROM the AutoArborist checkpoint
#
#   eval     eval_points_weak_validation.sbatch  (array 0-3, afterok stage 2)
#              scores every arm/split on the points VALIDATION split
#              (train.py already scored TEST at the end of each stage-2 run).
#
# Stage 1 is submitted per split -- never one array behind a single afterok -- so
# one split's failure cannot cancel the other (lesson from submit_box_ablation_v024.sh).
#
# Report the 2-arm Recall / mask-aware precision / F1 / counting MAE comparison per
# split x {test, validation} in notes/weak_supervision_points_table.md.
set -euo pipefail
cd /blue/ewhite/b.weinstein/src/MillionTrees

echo "=== quota check ==="
blue_quota || true
echo ""

SPLITS=(within-distribution out-of-distribution)

# --- stage 2 control: one array, no dependency ------------------------------
CONTROL=$(sbatch --parsable training/slurm/train_points_weak_control.sbatch)
echo "stage-2 control (both splits)        : $CONTROL"

declare -A PRE TREAT
for i in 0 1; do
  SPLIT=${SPLITS[$i]}

  PRE[$SPLIT]=$(sbatch --parsable --array=$i training/slurm/pretrain_points_autoarborist.sbatch)
  echo "stage-1 pretrain $SPLIT   : ${PRE[$SPLIT]} (array task $i)"

  TREAT[$SPLIT]=$(sbatch --parsable \
    --dependency=afterok:"${PRE[$SPLIT]}" \
    --export=ALL,SPLIT="$SPLIT" \
    training/slurm/train_points_weak_treatment.sbatch)
  echo "stage-2 treatment $SPLIT  : ${TREAT[$SPLIT]} (afterok ${PRE[$SPLIT]})"
done

# --- validation eval: afterok all four stage-2 jobs -------------------------
VALEVAL=$(sbatch --parsable \
  --dependency=afterok:"$CONTROL":"${TREAT[within-distribution]}":"${TREAT[out-of-distribution]}" \
  training/slurm/eval_points_weak_validation.sbatch)
echo "validation eval (array 0-3)          : $VALEVAL"

cat <<SUMMARY

Submitted. Write one job_ledger.md entry per job ID:
  $CONTROL                       stage-2 control, both splits
  ${PRE[within-distribution]}    stage-1 pretrain WD    -> blocks ${TREAT[within-distribution]}
  ${TREAT[within-distribution]}  stage-2 treatment WD   (afterok:${PRE[within-distribution]})
  ${PRE[out-of-distribution]}    stage-1 pretrain OOD   -> blocks ${TREAT[out-of-distribution]}
  ${TREAT[out-of-distribution]}  stage-2 treatment OOD  (afterok:${PRE[out-of-distribution]})
  $VALEVAL                       validation eval, all 4 arm/split combos (afterok stage 2)

FIRST CHECK on each log:
  pretrain  : "AutoArborist train images: ~22k", "loss-preset=pretrain APPLIED",
              BASELINE block prints finite keypoint_acc, "HF checkpoint exported".
  stage-2   : "deepforest branch OK: treeformer-training", "Loading pretrained
              TreeFormer checkpoint: <hf_checkpoint | weecology/...>", keypoint_acc
              not collapsing to 0.
  val eval  : per-source breakdown names Allen et al. 2025 AND Frey et al. 2026.
Then fill notes/weak_supervision_points_table.md.
SUMMARY
