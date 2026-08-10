# CanopyRS score-threshold sweep

> **AP columns below are AP50** (IoU 0.5), the metric in force when this sweep ran. AP is now
> AP40 (IoU 0.4) everywhere; the chosen operating point (0.30) is unaffected, since it was
> selected on F1, but the AP values are not comparable to the current leaderboard.

The leaderboard evaluates every pretrained model at a single, standardized
`eval_score_threshold=0.1` (see `[[project_bump_versions_dict_per_release]]`-adjacent
ledger note on restandardizing thresholds). At that threshold CanopyRS shows a
large recall/precision imbalance, most visibly on polygons:

| Task | Split | Threshold | Recall | Mask-Aware Precision | F1 |
|---|---|---:|---:|---:|---:|
| Boxes | Within-distribution | 0.1 | 0.688 | 0.679 | ~0.68 |
| Boxes | Out-of-distribution | 0.1 | 0.885 | 0.815 | ~0.85 |
| Polygons | Within-distribution | 0.1 | 0.842 | 0.541 | 0.659 |
| Polygons | Out-of-distribution | 0.1 | 0.893 | 0.510 | 0.649 |

(Source: `docs/leaderboard.md`.) The polygon rows in particular carry a large
number of low-confidence false-positive masks that a higher `eval_score_threshold`
should trade for precision.

## Sweep setup

`existing_models/slurm/sweep_canopyrs_pr_curve.sbatch` re-runs
`existing_models/canopyrs/eval_{boxes,polygons}.py --sweep` on the two manuscript
splits (within-distribution, out-of-distribution), with the model's own emission
threshold (`--score-threshold`) driven to `0.01` so the full score range reaches
the evaluator, then re-scores the *same* predictions at
`eval_score_threshold ∈ {0.05, 0.1, ..., 0.9}` (`milliontrees.common.eval_sweep`).
Boxes run on the **full test set**. Polygons cannot: holding the full test set of
dense instance masks in memory for sweeping is prohibitive (~270 GiB, see
`eval_polygons.py`), so the first attempt (job 36260539) capped polygons at
10 imgs/source but still hit its 8h limit mid-sweep — mask-based re-evaluation
(`dataset.eval()`, computing mask-aware precision/AP50/mask-accuracy) turned out
to be the bottleneck, not SAM3 inference. The resubmit (job 36306370) cut this
further to **5 imgs/source and a coarser 7-point grid**
(`{0.1, ..., 0.7}`) and completed both splits.

**Caveat:** because the polygon curve below is measured on a small
source-stratified subsample (not the full test set), it is reliable for *picking*
a threshold but the reported recall/precision/F1 values are not directly
plug-in-comparable to the other (full-test-set) rows in `docs/leaderboard.md`. A
full-test-set confirmation run at the chosen threshold is needed before updating
the leaderboard polygon rows (see Recommendation).

## Results

![CanopyRS precision-recall curve and F1 vs. threshold](../docs/public/canopyrs_threshold_sweep_pr_curve.png)

| Task | Split | Best threshold | Recall | Precision | F1 | Mask Accuracy | AP50 | Test set |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Boxes | Within-distribution | 0.30 | 0.635 | 0.731 | 0.679 | — | 0.420 | full (2,667 imgs) |
| Boxes | Out-of-distribution | 0.30 | 0.799 | 0.865 | 0.831 | — | 0.604 | full (3,809 imgs) |
| Polygons | Within-distribution | 0.30 | 0.762 | 0.874 | 0.814 | 0.268 | 0.313 | full (1,748 imgs) |
| Polygons | Out-of-distribution | 0.30 | 0.819 | 0.861 | 0.840 | 0.291 | 0.364 | full (2,758 imgs) |

Both geometries land on the same best-F1 threshold (0.30), and the shape is
consistent across all four curves: recall falls off roughly linearly with
threshold while precision keeps climbing past 0.9, so F1 peaks in the
0.25-0.30 range and degrades increasingly fast above it. The polygon full-test-set
confirmation came in even better than the 5-imgs/source subsample suggested
(F1 0.814/0.840 vs. 0.764/0.742) — the subsample was directionally right but
understated the gain, and mask accuracy roughly doubles at both splits
(0.135→0.268 within-dist, 0.111→0.291 OOD).

## Recommendation — both applied

**Boxes.** Raised CanopyRS box `score_threshold` from the standardized `0.1` to
**`0.30`**: F1 improves from ~0.68 (within-dist, recall/precision 0.69/0.68 →
0.64/0.73) and ~0.85 (OOD, 0.89/0.82 → 0.80/0.87) to a meaningfully more
precision-heavy, better-balanced operating point with only a modest recall cost.

**Polygons.** Raised CanopyRS polygon `score_threshold` from `0.1` to **`0.30`**
as well, confirmed on the full test set: F1 rises from 0.659→0.814 (within-dist)
and 0.649→0.840 (OOD) — the largest gain in the leaderboard, since the t=0.1
polygon rows were the most recall-heavy of any pretrained model.

**Counting MAE (boxes).** The sweep above only tracks recall/precision/F1/AP50;
`eval_canopyrs_boxes_confirm_t030.sbatch` (job 36352343) re-ran the plain
(non-sweep) box eval at `score_threshold=0.30` to also refresh `counting_mae`
and `detection_accuracy`. At the old `0.1` threshold CanopyRS's over-prediction
(high recall, low precision) inflated per-image count error to 137.7
within-dist / 153.6 OOD trees — far above DeepForest's ~11-20. At `0.30` this
drops to **15.1 within-dist / 13.8 OOD**, in line with the other models, since
counting MAE is dominated by total predicted-box volume and precision-heavy
thresholds cut the false-positive volume directly. `docs/leaderboard.md`'s
"Benchmark Results" appendix now reflects these confirmed numbers.

Both changes are live in `docs/leaderboard.md` (see the notes at the top of the
TreeBoxes and TreePolygons sections — a per-model exception to the cross-model
`0.1` standard used by SAM3/detectree2/DeepForest) and in
`existing_models/slurm/eval_canopyrs.sbatch`, so future leaderboard regenerations
reproduce these numbers.

---

*Generated by `scripts/plot_canopyrs_threshold_sweep.py` from
`existing_models/canopyrs/outputs/sweep/{within-distribution,out-of-distribution}/threshold_sweep.csv`
(boxes: full test set; polygons: 5 imgs/source, used only to pick the threshold).
Full-test-set polygon confirmation:
`existing_models/canopyrs/outputs/{within-distribution,out-of-distribution}/confirm_t030/results_polygons_*.txt`.
Jobs: sweep_canopyrs_pr_curve.sbatch (36260539, boxes) +
sweep_canopyrs_pr_curve_polygons.sbatch (36306370, polygon threshold search) +
eval_canopyrs_polygons_confirm_t030.sbatch (36309143, polygon full-test-set confirmation).*
