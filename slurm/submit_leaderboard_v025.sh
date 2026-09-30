#!/usr/bin/env bash
# Launch the COMPLETE v0.25 leaderboard refresh: fine-tuned training, pretrained
# existing models, the weak-supervision (Table 6) ablation, and the cross-geometry row.
#
#   bash slurm/submit_leaderboard_v025.sh            # gate on data readiness, then submit
#   bash slurm/submit_leaderboard_v025.sh --wait     # poll until ready, then submit
#   bash slurm/submit_leaderboard_v025.sh --dry-run  # print what would be submitted
#   bash slurm/submit_leaderboard_v025.sh --force    # skip the readiness gate
#
# Forked from slurm/submit_leaderboard_v024.sh with two corrections from user feedback
# on 2026-09-02 (memory feedback_maskrcnn_is_detectron_only), which the v0.24 script
# predates and never picked up:
#
#   1. The DeepForest (torchvision) Mask R-CNN polygon row is OUT of the manuscript.
#      "Mask R-CNN" means the native Detectron2 model only. This script does NOT submit
#      train_polygons_deepforest_annotationsafecrop.sbatch at all -- not as a leaderboard
#      row, and not as the Table 6 COCO baseline arm (see point 2).
#   2. The Table 6 weak-supervision ablation runs entirely on the DETECTRON2 stack via
#      training/weak_supervision/submit_d2_polygon_ablation.sh (3 arms: coco control /
#      box_pretrained_full (sequential) / cotrain, per split), not the DeepForest/
#      torchvision pretrain_backbone.sbatch + train_polygons_box_pretrained*.sbatch path
#      submit_leaderboard_v024.sh used.
#
# Still true from v0.24: submit_all.sh has drifted from the published table --
# scripts/make_benchmark_table.py reads specific output directories that submit_all.sh
# does not submit (see the v024 script's header for the row-by-row breakdown).
#
# The fine-tuned rows need no separate eval job: training/{boxes,points,polygons}/train.py
# each write results_<split>.txt into their own --output-dir at the end of training.
#
# Table 6 needs a SECOND stage this script does not submit: once the six D2 ablation
# training jobs below complete, a checkpoint-trajectory sweep (val-F1 selection) is
# required before the numbers are trustworthy -- see memory
# project_d2_weak_supervision_v023_done: last-iterate scoring is an early-stopping
# artifact on this dataset. Submit that sweep by hand once training finishes
# (eval_d2_polygon_wd_ckpt_sweep.sbatch / eval_d2_polygon_ood_ckpt_sweep.sbatch, or new
# array scripts sized to whatever checkpoints v0.25 retains).
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees

VERSION="${VERSION:-v0.25}"
LEDGER="${LEDGER:-/home/b.weinstein/logs/job_ledger.md}"
DRY_RUN=0; FORCE=0; WAIT=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --force)   FORCE=1 ;;
    --wait)    WAIT=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Submit time: the cross-geometry job uses it to reject checkpoints left in the
# (version-reused) countfix checkpoint dir by the previous dataset version.
SUBMIT_TIME=$(date '+%Y-%m-%dT%H:%M:%S')  # no space: it is passed through sbatch --export
export MIN_CKPT_TIME="$SUBMIT_TIME"

# ---------------------------------------------------------------- readiness gate ----
if [ "$FORCE" -eq 1 ]; then
  echo "!! --force: skipping the readiness gate. Half-written *_supervised_* image dirs"
  echo "!! surface as PIL.UnidentifiedImageError hours into a run."
elif [ "$WAIT" -eq 1 ]; then
  bash slurm/wait_for_release_data.sh wait "$VERSION"
else
  if ! bash slurm/wait_for_release_data.sh check "$VERSION"; then
    echo
    echo "$VERSION is not ready. Data lands ~14-18h after the packaging job starts;"
    echo "the zips take another ~24h and are NOT needed (notes/packaging_phase_timing.md)."
    echo "Re-run with --wait to poll, or --force to override."
    exit 1
  fi
fi

# --------------------------------------------------------- duplicate-job guard ------
# A same-name job already in the queue means a previous submit is still live; a second
# one shares the output dir and clobbers results.
SBATCHES=(
  training/slurm/train_boxes.sbatch
  training/slurm/train_points_896_countfix.sbatch
  training/slurm/train_polygons_detectron2.sbatch
  existing_models/slurm/eval_deepforest.sbatch
  existing_models/slurm/eval_treeformer_896_isolated.sbatch
  existing_models/slurm/eval_sam3.sbatch
  existing_models/slurm/eval_detectree2.sbatch
  existing_models/slurm/eval_canopyrs.sbatch
  training/slurm/pretrain_detectron2.sbatch
  training/slurm/train_polygons_d2_ablation.sbatch
  existing_models/slurm/eval_treeformer_sam2_crossgeom_v025.sbatch
  existing_models/slurm/eval_sam3_crossgeometry.sbatch
)
for f in "${SBATCHES[@]}"; do
  [ -f "$f" ] || { echo "missing sbatch script: $f" >&2; exit 1; }
done

RUNNING=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u || true)
CLASHES=""
# Read the job names off the scripts themselves rather than duplicating the list here,
# so renaming a #SBATCH --job-name can never silently disable this guard.
for f in "${SBATCHES[@]}"; do
  name=$(grep -m1 -- '--job-name=' "$f" | sed 's/.*--job-name=//' | awk '{print $1}')
  if [ -n "$name" ] && grep -qx "$name" <<< "$RUNNING"; then CLASHES="$CLASHES $name"; fi
done
if [ -n "$CLASHES" ]; then
  echo "!! Already queued/running:$CLASHES"
  echo "!! Cancel them or wait; duplicate submits share an output dir and clobber results."
  [ "$FORCE" -eq 1 ] || exit 1
fi

# ------------------------------------------------------------------- submit ---------
SUBMITTED=()

submit() {  # submit <ledger-why> <ledger-next> <sbatch args...>
  local why="$1" next="$2"; shift 2
  # Human-readable output goes to stderr so callers can capture the bare job id on
  # stdout without losing the log line.
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  DRY-RUN: sbatch $*" >&2
    echo "           Why: $why" >&2
    return 0
  fi
  local jid
  jid=$(sbatch --parsable "$@")
  echo "  submitted $jid: $*" >&2
  # CLAUDE.md section 1: a submit without a ledger entry is an incomplete submit.
  {
    printf '\n## %s — %s — %s\n' "$jid" "$(date '+%Y-%m-%d %H:%M')" "${*: -1}"
    printf 'Why: %s\n' "$why"
    printf 'Next: %s\n' "$next"
  } >> "$LEDGER"
  SUBMITTED+=("$jid")
  printf '%s' "$jid"
}

echo
echo "=== 1/4  Fine-tuned models (3 leaderboard rows x 2 splits) ==="

submit "$VERSION leaderboard: DeepForest RetinaNet box rows, both splits." \
       "results -> training/boxes/outputs/<split>/results_<split>.txt; feeds make_benchmark_table.py TreeBoxes finetuned row." \
       training/slurm/train_boxes.sbatch >/dev/null

# Points MUST run from the frozen .venv-treeformer. The shared .venv is pinned to the
# polygon DeepForest branch, on which point training silently trains to keypoint_acc=0
# instead of crashing. This sbatch hard-fails if the branch is wrong.
POINTS_JOB=$(submit \
  "$VERSION leaderboard: TreeFormer count-loss-fix point rows, both splits, frozen .venv-treeformer." \
  "results -> training/points/outputs/<split>_896_countfix/; array task 0 also gates the cross-geometry row." \
  training/slurm/train_points_896_countfix.sbatch)

submit "$VERSION leaderboard: Detectron2 Mask R-CNN polygon rows, both splits. The only polygon fine-tuned row in the manuscript (DeepForest/torchvision Mask R-CNN is out, feedback_maskrcnn_is_detectron_only)." \
       "results -> training/polygons/outputs/detectron2/<split>/results_<split>.txt." \
       training/slurm/train_polygons_detectron2.sbatch >/dev/null

echo
echo "=== 2/4  Pretrained existing models (8 leaderboard rows x 2 splits) ==="

submit "$VERSION leaderboard: DeepForest release weights, box task." \
       "results -> existing_models/deepforest/outputs/<split>/results_boxes_<split>.txt." \
       existing_models/slurm/eval_deepforest.sbatch >/dev/null

# The published TreeFormer row is the 896 isolated-venv eval, not eval_treeformer.sbatch.
submit "$VERSION leaderboard: TreeFormer release weights at image_size 896, frozen .venv-treeformer." \
       "results -> existing_models/treeformer/outputs/<split>_896_isolated/results_points_<split>.txt." \
       existing_models/slurm/eval_treeformer_896_isolated.sbatch >/dev/null

submit "$VERSION leaderboard: SAM3 zero-shot on all three geometries." \
       "results -> existing_models/sam3/outputs/<split>/results_{boxes,points,polygons}_<split>.txt (3 rows from one job)." \
       existing_models/slurm/eval_sam3.sbatch >/dev/null

submit "$VERSION leaderboard: detectree2 pretrained polygon model." \
       "results -> existing_models/detectree2/outputs/<split>/results_polygons_<split>.txt." \
       existing_models/slurm/eval_detectree2.sbatch >/dev/null

submit "$VERSION leaderboard: CanopyRS DINO Swin-L boxes + DINO/SAM3 polygons at the tuned 0.30 threshold. v0.24 was missing the OOD row on the published table despite existing on file (memory project_manuscript_missing_rows_v024) -- this submit covers both splits." \
       "results -> existing_models/canopyrs/outputs/<split>/results_{boxes,polygons}_<split>.txt (2 rows from one job). Confirm BOTH within-distribution and out-of-distribution land in docs tables this time." \
       existing_models/slurm/eval_canopyrs.sbatch >/dev/null

echo
echo "=== 3/4  Weak-supervision ablation (manuscript Table 6, Detectron2 stack) ==="

for SPLIT in within-distribution out-of-distribution; do
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  DRY-RUN: bash training/weak_supervision/submit_d2_polygon_ablation.sh $SPLIT" >&2
    continue
  fi
  echo "  --- $SPLIT ---" >&2
  D2_OUT=$(bash training/weak_supervision/submit_d2_polygon_ablation.sh "$SPLIT")
  echo "$D2_OUT" >&2
  PRETRAIN_ID=$(echo "$D2_OUT" | sed -n 's/^stage 1.*: \([0-9_]*\)$/\1/p')
  FREE_ID=$(echo "$D2_OUT"     | sed -n 's/^arms 0.*: \([0-9_]*\)$/\1/p')
  SEQ_ID=$(echo "$D2_OUT"      | sed -n 's/^arm 1.*: \([0-9_]*\)$/\1/p')
  {
    printf '\n## %s — %s — pretrain_detectron2.sbatch (SPLIT=%s, Table 6 stage 1)\n' \
      "$PRETRAIN_ID" "$(date '+%Y-%m-%d %H:%M')" "$SPLIT"
    printf 'Why: %s Table 6 stage 1: pretrain a native Detectron2 box backbone (MASK_ON=False) on TreeBoxes+unsupervised, %s split. Gates the box_pretrained_full sequential arm.\n' "$VERSION" "$SPLIT"
    printf 'Next: read mt_pretrain_d2_%s.{out,err}; require "Verified COCO init" + peak box_recall above the COCO baseline + a 295-tensor export. Unblocks the sequential arm below.\n' "$PRETRAIN_ID"
    printf '\n## %s — %s — train_polygons_d2_ablation.sbatch (SPLIT=%s, --array=0,2, coco+cotrain)\n' \
      "$FREE_ID" "$(date '+%Y-%m-%d %H:%M')" "$SPLIT"
    printf 'Why: %s Table 6 %s: COCO control (arm 0) + co-training with weak Weinstein tiles (arm 2). No stage-1 dependency.\n' "$VERSION" "$SPLIT"
    printf 'Next: results -> training/polygons/outputs/detectron2_ablation/%s/{coco,cotrain}/results_%s.txt. Do NOT report last-iterate -- v0.24 (job 40990287) showed the control peaks ep~10 then overfits monotonically to model_final. Run the checkpoint sweep and select on val F1 before filling Table 6.\n' "$SPLIT" "$SPLIT"
    printf '\n## %s — %s — train_polygons_d2_ablation.sbatch (SPLIT=%s, --array=1, sequential, afterok:%s)\n' \
      "$SEQ_ID" "$(date '+%Y-%m-%d %H:%M')" "$SPLIT" "$PRETRAIN_ID"
    printf 'Why: %s Table 6 %s: box-pretrained backbone (295-tensor stage-1 export) merged onto COCO, mask head stays COCO-init, fine-tuned on supervised polygons.\n' "$VERSION" "$SPLIT"
    printf 'Next: results -> training/polygons/outputs/detectron2_ablation/%s/box_pretrained_full/. Completes the six-cell (3 arms x 2 splits) v0.25 Table 6 once both splits finish. Then submit a val-F1 checkpoint-trajectory sweep (pattern: eval_d2_polygon_{wd,ood}_ckpt_sweep.sbatch) before writing the table.\n' "$SPLIT"
  } >> "$LEDGER"
  SUBMITTED+=("$PRETRAIN_ID" "$FREE_ID" "$SEQ_ID")
done

echo
echo "=== 4/4  Cross-geometry ==="

# Depends on array task 0 (within-distribution) only, so it starts without waiting for
# the out-of-distribution task. MIN_CKPT_TIME is exported above and inherited by sbatch.
submit "$VERSION leaderboard cross-geometry row: fine-tuned TreeFormer points -> SAM2 mask prompting, Run-M config, within-distribution checkpoint." \
       "results -> existing_models/treeformer_sam2/outputs/crossgeometry_v025/results_polygons_crossgeometry.txt; MIN_CKPT_TIME=$SUBMIT_TIME rejects stale pre-v0.25 checkpoints in the shared countfix dir." \
       --dependency="afterok:${POINTS_JOB}_0" \
       --export=ALL,MIN_CKPT_TIME="$SUBMIT_TIME" \
       existing_models/slurm/eval_treeformer_sam2_crossgeom_v025.sbatch >/dev/null

submit "$VERSION cross-geometry companion: SAM3 zero-shot polygons on the crossgeometry split (not a published row today; gives the cross-geometry comparison a pretrained baseline)." \
       "results -> existing_models/sam3/outputs/crossgeometry/results_polygons_crossgeometry.txt; add a RUNS entry in scripts/make_benchmark_table.py if it should be published." \
       existing_models/slurm/eval_sam3_crossgeometry.sbatch >/dev/null

# ------------------------------------------------------------------- summary --------
echo
if [ "$DRY_RUN" -eq 1 ]; then
  echo "DRY-RUN complete. Nothing was submitted and the ledger was not touched."
  exit 0
fi
echo "=== Submitted ${#SUBMITTED[@]} jobs for $VERSION: ${SUBMITTED[*]} ==="
echo "Ledger entries appended to $LEDGER"
echo
echo "Monitor:  squeue -u \$USER"
echo
echo "When everything is COMPLETED, rebuild the tables:"
echo "  uv run python scripts/make_benchmark_table.py --splits within-distribution out-of-distribution crossgeometry"
echo "  (DATA_VERSION in that script is already set to $VERSION)"
echo "Then submit the D2 checkpoint-trajectory sweep for Table 6 and fill"
echo "notes/d2_polygon_weak_supervision.md + manuscript Table 6."
