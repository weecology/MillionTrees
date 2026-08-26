# Detectron2 polygon weak-supervision ablation (v0.23 / AP40 + AP60)

**Status: running (jobs 40208343–40208348), results pending.** Design and plumbing are
settled and verified end-to-end on GPU by smoke job 40204749 (`SMOKE PASSED`); this file
gets its numbers when the runs land.

## The question

The DeepForest/torchvision box round finished on 2026-08-25 with a split result:

| Split | COCO control | Sequential pretrain | Co-training |
|---|---|---|---|
| within-distribution | 0.450 | 0.464 | **0.469** |
| out-of-distribution | **0.519** | 0.513 | 0.512 |

Co-training won within-distribution by +0.019 AP40 and *lost* out-of-distribution, which
retires the WD gain as an artifact — consistent with every earlier weak-supervision arm
being null WD and negative OOD.

But every arm in that round ran through DeepForest/torchvision, and DeepForest is the
weak implementation here: native Detectron2 more than doubled DeepForest's polygon AP50
on identical data. So the null is confounded with the stack. This ablation rebuilds all
three arms on **native Detectron2**, the model behind the published polygon rows, so a
null can be attributed to the weak labels rather than to DeepForest.

## The three arms

All three are Detectron2 `mask_rcnn_R_50_FPN_3x`, batch 8, lr 0.01, grad-clip 1.0,
image_size 448, `SCORE_THRESH_TEST` 0.15, scored by the MillionTrees streaming polygon
evaluator — i.e. leaderboard-comparable.

| Arm | Init | Train data |
|---|---|---|
| `coco` | released COCO Mask R-CNN, verified tensor-for-tensor (281 backbone/FPN) | supervised TreePolygons |
| `box_pretrained_full` | whole stage-1 network (295 tensors: backbone + FPN + RPN + ROI box head), mask head left at COCO | supervised TreePolygons |
| `cotrain` | COCO | supervised TreePolygons **+** the weak Weinstein tiles, one stage |

Stage 1 is native Detectron2 too (`training/polygons/pretrain_detectron2.py`): a Mask
R-CNN with `MASK_ON=False` trained on the unsupervised Weinstein et al. 2018 boxes
(41,691 tiles / 6.4M boxes WD; 46,823 / 7.1M OOD). Both stages therefore live in one
stack, with no cross-framework weight conversion to explain a result away.

## Design decisions worth remembering

**Matched compute, unlike the box round.** `--max-iter` is pinned identically across the
three arms of a split (WD 58,250; OOD 51,850 — 50 epochs' worth of the *control* arm's
data). The co-training arm's epoch is ~5x longer (9,317 polygon tiles vs 51,008 mixed),
so a shared "50 epochs" would have handed it 5x the updates. The box round had to carry
that as a caveat; here it is designed out.

**The mask head never sees a rectangle.** Box-only weak tiles have no polygon, so they
enter the co-training set with their bounding box as a stand-in `segmentation` and an
image-level `real_mask=False` flag. `MaskAwareROIHeads` drops those proposals before the
mask loss, so weak tiles contribute RPN and box losses only. Verified directly on CPU: a
batch of two box-only images gives `loss_mask` exactly 0.0 while `loss_cls`,
`loss_box_reg`, `loss_rpn_cls` and `loss_rpn_loc` stay non-zero; a mixed batch gives a
mask loss that differs from the all-polygon case.

**Both stage-1 defects from the box round are fixed by construction.** Gradient clipping
at norm 1.0 (without it the DeepForest stage 1 went NaN at lr 1e-2 and diverged at 1e-3),
and checkpoint selection on **holdout box recall** rather than loss (the val_loss-best
epoch of the earlier sweep exported a near-dead detector at recall 0.003 while a
worse-loss epoch detected ten times better).

**Init hygiene is asserted in both directions.** The control arm refuses to run unless all
281 backbone/FPN tensors match the released COCO weights — DeepForest's config default
silently substituted a NEON-trained checkpoint in two earlier ablations and invalidated
them. The sequential arm refuses to run if its backbone comes out byte-identical to COCO,
which would make it a control in disguise.

**One weak source, used identically by both arms.** `--weak-sources '*unsupervised*'`
selects Weinstein et al. 2018 only, for stage 1 *and* co-training, so the two arms differ
in *when* the weak labels are used, not *which* ones. The box round needed
`--exclude-sources '*Young*'` to achieve this; here Young et al. 2025 ("weak supervised",
not "unsupervised") is never selected in the first place. It is also excluded from the
stage-1 selection holdout, which must be human boxes — scoring against pseudo-labels
would measure agreement with another model rather than recall of real trees.

## Caveats to carry with the numbers

- **No early stopping.** The driver scores `model_final.pth`, the last iterate. That is
  the recipe behind the published Detectron2 polygon rows and it is identical across
  arms, so the comparison is controlled, but it is not best-checkpoint selection. If the
  D2 polygon model turns out to overfit the way the torchvision one does
  (`project_polygon_maskrcnn_overfits`), this is the first thing to revisit.
- **Domain overlap, not leakage.** No weak tile shares a filename or a scene stem with
  any polygon test or validation tile (checked for both splits). But the weak tiles are
  NEON imagery and the polygon test set includes `NEON MultiTemporal` and
  `NEON combined crowns`, so the weak set is not domain-disjoint from the test set. This
  applies equally to arms 1 and 2, so the between-arm comparison is unaffected.
- Numbers here are AP40 + AP60 on v0.23 and are **not** comparable to the 2026-07-01 D2
  Table-6 rows, which were v0.20-era AP50 and used a trunk-only (265-tensor) transfer.

## Trap found while building this

`--mini` does not make a polygon run small. All three loaders pick the mini versions dict
on `mini=True` and then, a dozen lines later, the "select supervised-only dataset by
default" branch overwrites `_dataset_name` with `TreePolygons_supervised` /
`TreeBoxes_supervised` / `TreePoints_supervised`. Since `include_unsupervised=False` is the
default, **`mini=True` is silently discarded** and the loader lands on the full release.
Job 40200530 timed out at 1h because of it: the 10-iteration training step was instant and
the evaluation over 1,742 full-release test tiles ate the wall clock. This is pre-existing
and project-wide — `train_polygons_detectron2_smoke.sbatch` has been running full-release
"smokes" all along. Not fixed here (a shipped-loader change affects every caller); the
smoke gets its smallness from `--eval-split validation` (59 tiles) and `--max-iter` instead.

## Files

- `training/polygons/d2_weak_supervision.py` — TreeBoxes→D2 bridge, `MaskAwareROIHeads`,
  `WeakMaskDatasetMapper`, COCO-init verification, whole-network export/merge.
- `training/polygons/pretrain_detectron2.py` — stage 1.
- `training/polygons/train_detectron2.py` — stage 2, `--init-mode {coco,box_pretrained_full,cotrain}`.
- `training/slurm/pretrain_detectron2.sbatch`, `training/slurm/train_polygons_d2_ablation.sbatch`,
  `training/slurm/pretrain_d2_polygon_ablation_smoke.sbatch`.
- `training/weak_supervision/submit_d2_polygon_ablation.sh` — one split per invocation.

## Jobs

| Job | Split | What |
|---|---|---|
| 40208343 | within-distribution | stage 1 (weak-box pretraining) |
| 40208344 | within-distribution | arms 0 (control) + 2 (co-training) |
| 40208345 | within-distribution | arm 1 (sequential), `afterok:40208343` |
| 40208346 | out-of-distribution | stage 1 |
| 40208347 | out-of-distribution | arms 0 + 2 |
| 40208348 | out-of-distribution | arm 1, `afterok:40208346` |

`MAX_ITER` is 58,250 (WD) / 51,850 (OOD) on all three arms of a split.

## Results

_Pending._ Fill Recall / mask-aware Precision / AP40 / AP60 / worst-group AP40 per arm
per split from
`training/polygons/outputs/detectron2_ablation/<split>/{coco,box_pretrained_full,cotrain}/results_<split>.txt`,
then compare against the DeepForest box round in the table at the top of this file.

| Split | Arm | Recall | Mask-aware P | AP40 | AP60 | Worst-group AP40 |
|---|---|---|---|---|---|---|
| within-distribution | COCO control | | | | | |
| within-distribution | Sequential pretrain | | | | | |
| within-distribution | Co-training | | | | | |
| out-of-distribution | COCO control | | | | | |
| out-of-distribution | Sequential pretrain | | | | | |
| out-of-distribution | Co-training | | | | | |
