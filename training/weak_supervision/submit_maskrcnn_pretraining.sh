#!/bin/bash
# Two-stage Mask R-CNN weak-supervision ablation (the polygon whole-network arm).
#
# Stage 1 pretrains the POLYGON architecture on the unsupervised boxes with the
# mask branch off; stage 2 fine-tunes on polygons from that 404-tensor export and
# against a paired COCO control. See the two sbatch files for the full rationale.
#
# The splits are submitted as INDEPENDENT chains on purpose. The original
# submit_box_pretraining_ablation_full.sh put a whole array behind one afterok,
# so a single task's quota death cancelled both splits. Here each split's stage 2
# depends only on its own stage 1, and a failure in one leaves the other running.
#
# The COCO control does NOT depend on stage 1 -- it has nothing to wait for, and
# holding it in the queue for a 24h pretrain would waste a day for no gain. It is
# still a paired control: same submission, same code state, same recipe. That is
# what the box-side replicate spread (AP40 0.476 vs 0.450 on an identical config)
# showed was missing.
set -euo pipefail
cd "$(dirname "$0")/../.."

SPLITS=("${@:-within-distribution out-of-distribution}")
[ $# -eq 0 ] && SPLITS=(within-distribution out-of-distribution)

for SPLIT in "${SPLITS[@]}"; do
  # Guard against the duplicate-submit failure mode: two jobs on the SAME split
  # share an output directory and clobber each other's exports. The name has to
  # carry the split, otherwise submitting the second split always trips the
  # check -- and a guard that always fires is one people learn to ignore.
  JOB_NAME="mt_pretrain_maskrcnn_${SPLIT}"
  if squeue -u "$USER" -h -o "%j" | grep -qx "$JOB_NAME"; then
    echo "WARNING: $JOB_NAME is already queued/running; it writes to the same" >&2
    echo "         output directory. Not submitting a duplicate for $SPLIT." >&2
    continue
  fi

  STAGE1=$(sbatch --parsable --job-name="$JOB_NAME" \
    --export=ALL,SPLIT="$SPLIT" \
    training/slurm/pretrain_maskrcnn_for_polygons.sbatch)
  echo "[$SPLIT] stage 1 (pretrain maskrcnn on weak boxes): $STAGE1"

  CTRL=$(sbatch --parsable --array=0 \
    --export=ALL,SPLIT="$SPLIT" \
    training/slurm/train_polygons_maskrcnn_pretrained.sbatch)
  echo "[$SPLIT] stage 2 COCO control (no dependency):      $CTRL"

  # Arms 1 and 2 both consume stage-1 exports, so both wait on it. They are one
  # array because they are the same dependency, not because they are the same
  # experiment: arm 1 is the treatment, arm 2 is the transfer-depth control that
  # keeps a win attributable.
  TREAT=$(sbatch --parsable --array=1-2 \
    --dependency=afterok:"$STAGE1" \
    --export=ALL,SPLIT="$SPLIT" \
    training/slurm/train_polygons_maskrcnn_pretrained.sbatch)
  echo "[$SPLIT] stage 2 arms 1-2 (afterok:$STAGE1):         $TREAT"
  echo
done

cat <<'MSG'
Remember (CLAUDE.md section 1): append one ledger entry per job ID to
/home/b.weinstein/logs/job_ledger.md with Why / Next before moving on.
MSG
