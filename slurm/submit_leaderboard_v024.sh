#!/usr/bin/env bash
# Launch the COMPLETE v0.24 leaderboard refresh: fine-tuned training, pretrained
# existing models, the weak-supervision (Table 6) ablation, and the cross-geometry row.
#
#   bash slurm/submit_leaderboard_v024.sh            # gate on data readiness, then submit
#   bash slurm/submit_leaderboard_v024.sh --wait     # poll until ready, then submit
#   bash slurm/submit_leaderboard_v024.sh --dry-run  # print what would be submitted
#   bash slurm/submit_leaderboard_v024.sh --force    # skip the readiness gate
#
# WHY THIS EXISTS INSTEAD OF slurm/submit_all.sh
# ---------------------------------------------
# submit_all.sh has drifted from the published table. scripts/make_benchmark_table.py
# reads specific output directories, and three of them are produced by sbatch scripts
# that submit_all.sh does not submit:
#
#   row                          table reads                                    submit_all.sh submits
#   TreeFormer (finetuned)       points/outputs/{split}_896_countfix            train_points.sbatch -> {split}/        WRONG
#   Mask R-CNN (Detectron2)      polygons/outputs/detectron2/{split}            train_polygons.sbatch -> {split}/      MISSING
#   DeepForest Mask R-CNN        polygons/outputs/deepforest_annotationsafecrop train_polygons.sbatch -> {split}/      MISSING
#   TreeFormer (release)         treeformer/outputs/{split}_896_isolated        eval_treeformer.sbatch -> {split}/     WRONG
#
# Running submit_all.sh would burn GPU-days and leave the table showing '-' for those
# rows. This script submits the table-canonical set instead.
#
# DEDUPLICATION: train_polygons_deepforest_annotationsafecrop.sbatch is BOTH the
# "DeepForest Mask R-CNN" leaderboard row AND the COCO baseline arm of the Table 6
# weak-supervision comparison. submit_weak_supervision.sh submits it a second time.
# Two jobs writing one output dir clobber each other, so it is submitted ONCE here and
# the weak-supervision arm reuses it.
#
# The fine-tuned rows need no separate eval job: training/{boxes,points,polygons}/train.py
# each write results_<split>.txt into their own --output-dir at the end of training.
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees

VERSION="${VERSION:-v0.24}"
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
  training/slurm/train_polygons_deepforest_annotationsafecrop.sbatch
  existing_models/slurm/eval_deepforest.sbatch
  existing_models/slurm/eval_treeformer_896_isolated.sbatch
  existing_models/slurm/eval_sam3.sbatch
  existing_models/slurm/eval_detectree2.sbatch
  existing_models/slurm/eval_canopyrs.sbatch
  training/slurm/pretrain_backbone.sbatch
  training/slurm/train_polygons_box_pretrained_annotationsafecrop.sbatch
  existing_models/slurm/eval_treeformer_sam2_crossgeom_v024.sbatch
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
echo "=== 1/4  Fine-tuned models (4 leaderboard rows x 2 splits) ==="

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

submit "$VERSION leaderboard: Detectron2 Mask R-CNN polygon rows, both splits." \
       "results -> training/polygons/outputs/detectron2/<split>/results_<split>.txt." \
       training/slurm/train_polygons_detectron2.sbatch >/dev/null

# Dual purpose: leaderboard polygon row AND the Table 6 COCO baseline arm.
submit "$VERSION leaderboard DeepForest Mask R-CNN polygon rows AND the COCO baseline arm of the Table 6 weak-supervision comparison (submitted once, not twice)." \
       "results -> training/polygons/outputs/deepforest_annotationsafecrop/<split>/; compare against the box-pretrained arm for Table 6." \
       training/slurm/train_polygons_deepforest_annotationsafecrop.sbatch >/dev/null

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

submit "$VERSION leaderboard: CanopyRS DINO Swin-L boxes + DINO/SAM3 polygons at the tuned 0.30 threshold." \
       "results -> existing_models/canopyrs/outputs/<split>/results_{boxes,polygons}_<split>.txt (2 rows from one job)." \
       existing_models/slurm/eval_canopyrs.sbatch >/dev/null

echo
echo "=== 3/4  Weak-supervision ablation (manuscript Table 6) ==="

PRETRAIN_JOB=$(submit \
  "$VERSION Table 6 stage 1: pretrain a box backbone on TreeBoxes+unsupervised, both splits." \
  "backbone -> training/weak_supervision/outputs/pretrain/<split>/box_backbone_<split>.pt; gates the box-pretrained polygon arm." \
  training/slurm/pretrain_backbone.sbatch)

submit "$VERSION Table 6 stage 2: polygon Mask R-CNN initialised from the box-pretrained backbone (annotationsafecrop), vs the COCO arm submitted above." \
       "results -> training/weak_supervision/outputs/polygon_box_pretrained_annotationsafecrop/<split>/; write the 4-row Recall/Mask-aware-Precision/AP40 table to notes/weak_supervision_pretraining_table.md." \
       --dependency="afterok:$PRETRAIN_JOB" \
       training/slurm/train_polygons_box_pretrained_annotationsafecrop.sbatch >/dev/null

echo
echo "=== 4/4  Cross-geometry ==="

# Depends on array task 0 (within-distribution) only, so it starts without waiting for
# the out-of-distribution task. MIN_CKPT_TIME is exported above and inherited by sbatch.
submit "$VERSION leaderboard cross-geometry row: fine-tuned TreeFormer points -> SAM2 mask prompting, Run-M config, within-distribution checkpoint." \
       "results -> existing_models/treeformer_sam2/outputs/crossgeometry_v024/results_polygons_crossgeometry.txt; MIN_CKPT_TIME=$SUBMIT_TIME rejects stale v0.23 checkpoints in the shared countfix dir." \
       --dependency="afterok:${POINTS_JOB}_0" \
       --export=ALL,MIN_CKPT_TIME="$SUBMIT_TIME" \
       existing_models/slurm/eval_treeformer_sam2_crossgeom_v024.sbatch >/dev/null

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
echo "Then refresh Table 6 in notes/weak_supervision_pretraining_table.md."
