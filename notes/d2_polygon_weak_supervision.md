# Detectron2 polygon weak-supervision ablation (AP40 + AP60)

**Status: v0.24 OOD re-run done + checkpoint-trajectory sweep done (2026-09-03). The
headline result is a NEGATIVE: the weak-supervision "gains" in the last-iterate numbers
are early-stopping artifacts. See "Results" below.** v0.23 first ran as jobs 40208343–48
(2026-08-25); v0.24 OOD re-ran as 40874559/40874560 (2026-09-02, WD re-run 40874556/57
was cancelled and not resubmitted); the per-epoch checkpoint sweep is job 40990287
(2026-09-03, OOD only, `coco` + `box_pretrained_full`, inference only).

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

### Last-iterate numbers (`model_final.pth`, the recipe behind the published D2 rows)

F1 = harmonic mean of mask-recall and mask-aware precision (the leaderboard convention).

| Split | Arm | Recall | Mask-aware P | F1 | AP40 | AP60 | wg AP40 |
|---|---|---|---|---|---|---|---|
| within-distribution (v0.23) | COCO control        | 0.654 | 0.924 | 0.766 | 0.453 | 0.297 | 0.012 |
| within-distribution (v0.23) | Sequential pretrain | 0.655 | 0.921 | 0.766 | 0.458 | 0.296 | 0.021 |
| within-distribution (v0.23) | Co-training         | 0.759 | 0.817 | 0.787 | 0.380 | 0.252 | 0.001 |
| out-of-distribution (v0.24) | COCO control        | 0.376 | 0.943 | 0.538 | 0.243 | 0.188 | 0.010 |
| out-of-distribution (v0.24) | Sequential pretrain | 0.402 | 0.938 | 0.563 | 0.275 | 0.212 | 0.014 |
| out-of-distribution (v0.24) | Co-training         | 0.686 | 0.790 | 0.734 | 0.302 | 0.225 | 0.076 |

Read at face value this says co-training wins OOD F1 by +0.20 and sequential adds +0.03
AP40. **Both readings are wrong** — see the sweep.

### The checkpoint-trajectory sweep kills the result (job 40990287, OOD, 2026-09-03)

Every retained checkpoint of the OOD `coco` and `box_pretrained_full` runs re-scored with
the MillionTrees streaming evaluator (inference only, no retraining). `iters/epoch = 863`
(6902 imgs / 8); `SOLVER.STEPS = (36295, 46665)` so LR is 0.01 through it 34519, then
1e-3, then 1e-4.

**COCO control:**

| epoch | iter | LR | recall | mask-P | F1 | AP40 | AP60 | wg AP40 |
|---|---|---|---|---|---|---|---|---|
| 5   | 4314  | 1e-2 | 0.675 | 0.811 | 0.737 | 0.361 | 0.280 | 0.008 |
| **10**  | 8629  | 1e-2 | 0.735 | 0.786 | **0.760** | **0.509** | **0.396** | 0.069 |
| 15  | 12944 | 1e-2 | 0.619 | 0.884 | 0.728 | 0.417 | 0.321 | 0.015 |
| 20  | 17259 | 1e-2 | 0.631 | 0.861 | 0.728 | 0.424 | 0.317 | 0.014 |
| 25  | 21574 | 1e-2 | 0.567 | 0.880 | 0.690 | 0.342 | 0.251 | 0.008 |
| 30  | 25889 | 1e-2 | 0.528 | 0.887 | 0.662 | 0.366 | 0.269 | 0.016 |
| 35  | 30204 | 1e-2 | 0.480 | 0.923 | 0.632 | 0.288 | 0.209 | 0.009 |
| 40  | 34519 | 1e-2 | 0.530 | 0.920 | 0.673 | 0.372 | 0.275 | 0.017 |
| 45  | 38834 | 1e-3 | 0.433 | 0.933 | 0.591 | 0.291 | 0.222 | 0.010 |
| 50  | 43149 | 1e-3 | 0.429 | 0.942 | 0.590 | 0.287 | 0.222 | 0.010 |
| 55  | 47464 | 1e-4 | 0.397 | 0.941 | 0.558 | 0.265 | 0.203 | 0.010 |
| 60  | 51779 | 1e-4 | 0.391 | 0.941 | 0.552 | 0.258 | 0.196 | 0.010 |
| final | 51850 | 1e-4 | 0.376 | 0.943 | 0.538 | 0.243 | 0.188 | 0.010 |

**Sequential (`box_pretrained_full`):**

| epoch | iter | LR | recall | mask-P | F1 | AP40 | AP60 | wg AP40 |
|---|---|---|---|---|---|---|---|---|
| 5   | 4314  | 1e-2 | 0.672 | 0.822 | 0.739 | 0.455 | 0.357 | 0.019 |
| 10  | 8629  | 1e-2 | 0.731 | 0.806 | 0.767 | **0.507** | 0.391 | 0.043 |
| 15  | 12944 | 1e-2 | 0.706 | 0.833 | 0.764 | 0.470 | 0.366 | 0.028 |
| **20**  | 17259 | 1e-2 | 0.694 | 0.869 | **0.772** | 0.489 | 0.378 | 0.046 |
| 25  | 21574 | 1e-2 | 0.568 | 0.892 | 0.694 | 0.367 | 0.265 | 0.037 |
| 30  | 25889 | 1e-2 | 0.552 | 0.906 | 0.686 | 0.351 | 0.259 | 0.022 |
| 35  | 30204 | 1e-2 | 0.603 | 0.891 | 0.719 | 0.424 | 0.317 | 0.021 |
| 40  | 34519 | 1e-2 | 0.633 | 0.852 | 0.726 | 0.435 | 0.329 | 0.033 |
| 45  | 38834 | 1e-3 | 0.439 | 0.940 | 0.598 | 0.299 | 0.233 | 0.015 |
| 50  | 43149 | 1e-3 | 0.390 | 0.946 | 0.552 | 0.264 | 0.203 | 0.010 |
| 55  | 47464 | 1e-4 | 0.400 | 0.937 | 0.561 | 0.272 | 0.210 | 0.014 |
| 60  | 51779 | 1e-4 | 0.402 | 0.937 | 0.563 | 0.274 | 0.211 | 0.014 |
| final | 51850 | 1e-4 | 0.402 | 0.938 | 0.563 | 0.275 | 0.212 | 0.014 |

### What this means

1. **Both arms peak around epoch 10–20 and then overfit for 40+ epochs.** Peak vs
   last-iterate: `coco` F1 0.760 → 0.538, AP40 0.509 → 0.243; `box_pretrained_full`
   F1 0.772 → 0.563, AP40 0.507 → 0.275. Last-iterate scoring throws away **~40–50 % of
   AP40**. The decay is overfitting at full LR (recall 0.735 → 0.48 by ep 35, precision
   creeping 0.79 → 0.92 as the model turns conservative); the LR drops at ep 42/54 don't
   cause the collapse, they just fail to arrest it.

2. **At matched (peak) checkpoint selection, sequential weak-box pretraining buys nothing.**
   `coco` peak F1 0.760 / AP40 0.509 vs `box_pretrained_full` peak F1 0.772 / AP40 0.507 —
   inside run-to-run noise. The +0.03 AP40 "gain" at last-iterate is just that the
   pretrained arm overfits a hair slower.

3. **Co-training's apparent OOD F1 win (0.734 vs 0.538) is an early-stopping artifact.**
   Co-training's mixed set is ~87 % weak box tiles, so 51 850 iters is only ~7.7 passes
   over the polygon data (`n_train` 53 725) vs the control's ~60. It "wins" only because
   its diluted schedule accidentally stops near a reasonable point while the controls are
   scored 50 epochs past their peak. A properly checkpoint-selected COCO control
   (F1 0.760 / AP40 0.509) **beats** co-training (0.734 / 0.302) outright — and on AP40 by
   0.20. Co-training's one real edge is worst-group AP40 (0.076 vs ~0.010), but that is on
   a near-zero baseline (one OOD source never works for any arm).

### Consequence for manuscript Table 6

Every row was scored at `model_final.pth`, i.e. at the overfit endpoint, and the arms
overfit at different rates, so the table currently measures overfitting rate, not weak
supervision. **Table 6 must be regenerated with validation-F1 (or val-AP40) checkpoint
selection on every arm** before any weak-supervision claim. Expectation after that fix:
the three arms collapse to roughly F1 0.76 / AP40 0.50 on OOD and the ablation reports a
clean null. (Caveat: the peaks above are read off the *test* set — an optimistic bound;
real selection needs a validation curve. But the early-peak-then-decay shape is identical
across both swept arms and unambiguous.)

Mechanism to add: `train_detectron2.py` currently scores only `model_final.pth`
(line ~497). Needs a `--eval-split validation` pass per retained checkpoint (or an
in-training hook) picking the best by val F1, then a single test eval of that checkpoint.
`select_checkpoint.py` does this for the DeepForest polygon stack already.

Per-checkpoint prediction dumps are at
`training/polygons/outputs/detectron2_ablation/out-of-distribution/{coco,box_pretrained_full}_ckpt_sweep/ep*/preds.pkl`
for offline score-threshold / IoU re-scoring.
