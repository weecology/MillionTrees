#!/usr/bin/env bash
# Recovery submit for the 2026-08-11 disk-quota wipeout.
#
# On 2026-08-11 the ewhite group hit its 33T /blue quota. Four jobs died of it, three of
# them after burning most of their GPU time, and one dependent job was cancelled as
# collateral:
#
#   39193929_1  pretrain full, OOD   trained all 20 epochs (5h34m), then torch.save() of
#                                    the export truncated at 954 KB of 94 MB. No
#                                    ModelCheckpoint (the unsupervised-only subset leaves
#                                    the test split empty), so nothing is recoverable.
#                                    The truncated file has been renamed
#                                    box_backbone_out-of-distribution.pt.TRUNCATED_39193929
#                                    and is provably unloadable.
#   39193930    box train full, both CANCELLED — afterok on the whole 2-task pretrain
#                                    array, so task 1's failure killed BOTH splits even
#                                    though the within-distribution network exported fine.
#   39185675_1  polygon COCO, OOD    trained 20 epochs (10h17m), died writing an eval
#                                    overlay PNG. Checkpoint survived -> eval-only recovery.
#   39185676_*  polygon boxinit      died at ~1 min writing deepforest_train.csv. Nothing
#                                    to salvage, but nothing was lost either.
#
# CHECK BEFORE RUNNING THIS: blue_quota. The ewhite group was at 31.94T/33T on 2026-08-12.
# These jobs write ~1.5T of checkpoints, annotation CSVs and viz PNGs between them; if the
# group is back near the limit they will die the same way. `find training -name '*.ckpt'`
# was 62.5 GB across 181 files at last count, most of it superseded runs.
set -euo pipefail

cd /blue/ewhite/b.weinstein/src/MillionTrees

echo "=== quota check (do not proceed if headroom is under ~2T) ==="
blue_quota || true
echo ""

# --- Box -> box whole-network ablation -------------------------------------------------
# Split the two array tasks apart so one split's failure cannot cancel the other again.

echo "[1/5] Box whole-network TRAIN, within-distribution (no dependency — its 301-tensor"
echo "      network from 39193929_0 already exists)..."
BOX_WD=$(sbatch --parsable --array=0 training/slurm/train_boxes_box_pretrained_full_v023.sbatch)
echo "      job $BOX_WD"

echo "[2/5] Box whole-network PRETRAIN, out-of-distribution only (full 5.5h re-run)..."
PRETRAIN_OOD=$(sbatch --parsable --array=1 training/slurm/pretrain_backbone_v023_cocoinit_full.sbatch)
echo "      job $PRETRAIN_OOD"

echo "[3/5] Box whole-network TRAIN, out-of-distribution (afterok:$PRETRAIN_OOD)..."
BOX_OOD=$(sbatch --parsable --array=1 --dependency=afterok:"$PRETRAIN_OOD" \
  training/slurm/train_boxes_box_pretrained_full_v023.sbatch)
echo "      job $BOX_OOD"

# --- Polygon Table 6 ablation ----------------------------------------------------------

echo "[4/5] Polygon COCO control, out-of-distribution — EVAL ONLY from the surviving"
echo "      epoch-19 checkpoint (~0.5h instead of a 10h retrain)..."
POLY_EVAL=$(sbatch --parsable training/slurm/eval_polygons_coco_v023_ood_recover.sbatch)
echo "      job $POLY_EVAL"

echo "[5/5] Polygon box-pretrained treatment, both splits (full re-run; reads the"
echo "      trunk-only backbones from 39129719, which are intact)..."
POLY_BOX=$(sbatch --parsable training/slurm/train_polygons_box_pretrained_v023_cleaninit.sbatch)
echo "      job $POLY_BOX"

cat <<SUMMARY

Submitted. Write one ledger entry per job ID in /home/b.weinstein/logs/job_ledger.md:
  $BOX_WD        box whole-network train, within-distribution
  $PRETRAIN_OOD  box whole-network pretrain, OOD          -> blocks $BOX_OOD
  $BOX_OOD       box whole-network train, OOD             (afterok:$PRETRAIN_OOD)
  $POLY_EVAL     polygon COCO control OOD, eval-only recovery
  $POLY_BOX      polygon box-pretrained, both splits

FIRST CHECK on each log:
  box train      "Loaded FULL pretrained network: 301 tensors (trunk 265, FPN 16, heads 20)"
                 — if it says "Loaded 265 backbone keys" the truncation is back.
  box pretrain   "Verified COCO init: all 281 backbone/FPN tensors match torchvision
                 RetinaNet COCO_V1", then "Full network export written to ... (301 tensors)".
  polygon train  "Verified COCO init: all 366 backbone/FPN tensors match torchvision Mask
                 R-CNN v2 COCO_V1", then "Loaded 265 backbone keys; 98 keys left at COCO init."

Then fill in notes/weak_supervision_pretraining_table.md sections 1-3, which already carry
the finished trunk-only box comparison and the one finished polygon arm.
SUMMARY
