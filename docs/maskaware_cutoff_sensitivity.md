# How sensitive is F1 to the mask-aware precision cutoff?

Follow-up to `docs/validation_ap_completeness.md`, which attributed 41% of the F1-vs-AP50 gap to
`MaskAwareDetectionPrecision` "forgiving any unmatched prediction on ≥50% canopy pixels". This
sweeps that cutoff (`tree_fraction_threshold`) over its whole range on the v0.22 **validation**
split, re-scoring the saved prediction dumps in `outputs/validation_preds/` — no model was re-run.

Reproduce with:

```bash
python scripts/sweep_maskaware_precision.py \
  --predictions outputs/validation_preds/canopyrs_boxes_validation.pkl ... \
  --data-dir /orange/ewhite/web/public/MillionTrees/TreeBoxes_v0.22 \
  --out-dir docs/public/maskaware_cutoff
```

Recall, matching (IoU 0.4) and score thresholds are untouched — **only the canopy fraction a
false positive needs in order to be waived changes.** `off` means no forgiveness at all, i.e.
plain precision. Double detections (unmatched predictions that still clear IoU 0.4 against some
ground-truth tree) are never forgiven at any cutoff, matching the metric.

## Result

| Model | Task | Recall | P@0.0 | P@0.25 | P@0.5 (leaderboard) | P@0.75 | P@1.0 | P off | F1@0.5 | F1 off | FPs forgiven @0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CanopyRS-DINO-SwinL | TreeBoxes | 0.549 | 0.915 | 0.911 | 0.887 | 0.836 | 0.743 | 0.508 | 0.678 | 0.528 | 1914/2186 (88%) |
| DeepForest-pretrained | TreeBoxes | 0.287 | 1.000 | 0.984 | 0.967 | 0.924 | 0.796 | 0.279 | 0.443 | 0.283 | 3644/3676 (99%) |
| DeepForest-finetuned | TreeBoxes | 0.328 | 1.000 | 0.842 | 0.818 | 0.785 | 0.697 | 0.204 | 0.469 | 0.252 | 5348/5866 (91%) |
| SAM3 | TreeBoxes | 0.538 | 0.645 | 0.631 | 0.613 | 0.582 | 0.475 | 0.278 | 0.573 | 0.367 | 4407/6057 (73%) |
| Mask R-CNN (finetuned) | TreePolygons | 0.430 | 0.986 | 0.975 | 0.973 | 0.965 | 0.885 | 0.385 | 0.597 | 0.406 | 3682/3728 (99%) |
| Detectree2 | TreePolygons | 0.460 | 0.712 | 0.639 | 0.631 | 0.624 | 0.581 | 0.250 | 0.532 | 0.324 | 5386/6538 (82%) |

Full sweep (21 cutoffs × per-source rows): `docs/public/maskaware_cutoff/maskaware_cutoff_sweep.csv`
and `docs/public/maskaware_cutoff_polygons/maskaware_cutoff_sweep.csv`.

## Reading of the result

**The cutoff *value* barely matters; having forgiveness at all matters enormously.** Moving 0.5 →
0.75 costs CanopyRS 0.015 F1; turning forgiveness off costs it 0.150. Across the whole 0.0 → 1.0
range no model's F1 moves more than 0.08, and then it falls 0.15–0.22 at `off`. The reason is in
the right panel of the per-model figure: 70% of CanopyRS's forgivable false positives sit on
**100%** canopy pixels (91% for DeepForest-pretrained), so no cutoff below 1.0 can charge them.
Tuning the cutoff is therefore not a lever — the metric is effectively binary on this imagery.

**The forgiveness is doing most of the work in the reported precision.** At the leaderboard
setting, 73–99% of unmatched predictions are waived. DeepForest-pretrained's headline precision
of 0.967 comes from waiving 3644 of 3676 false positives; charge them and it is 0.279.

**It reverses precision rankings.** Mask-aware precision ranks DeepForest-pretrained (0.967)
*above* CanopyRS (0.887); plain precision ranks them 0.279 vs 0.508. A model that blankets canopy
with boxes is rewarded exactly where the metric is most forgiving. The overall F1 *ordering*
survives (only the two DeepForest variants swap), but the margins do not: CanopyRS's F1 lead over
Mask R-CNN grows from 0.081 to 0.122 once forgiveness is off.

**Low-canopy tiles are where the cutoff is visible at all.** On `FIN09` (100% canopy) precision is
0.98 at every cutoff up to 1.0 and 0.58 with forgiveness off. On `SPA32` (40% canopy) the same
sweep runs 0.56 → 0.31, because there is bare ground for a false positive to land on.

## Figures

Per model (`docs/public/maskaware_cutoff/`, polygons in `docs/public/maskaware_cutoff_polygons/`):

- `*_cutoff_sweep.png` — left: precision / F1 / recall vs the cutoff, with the `off` point;
  right: the canopy-cover distribution of the forgivable false positives, which *is* the
  sensitivity curve (the share forgiven at cutoff *t* is the mass at or above *t*).
- `*_overlay_<tile>.png` — the same tile scored at cutoff 0.5, 0.9 and off. White = ground truth,
  blue = matched, dashed green = forgiven because it sits on canopy, orange = charged as a false
  positive. Non-canopy pixels are washed out, so the region where forgiveness is unavailable is
  visible. Tiles are chosen to span the canopy-cover range (most / median / least canopied).
- `maskaware_cutoff_f1_all_models.png` — F1 vs cutoff for every model on one axes.

## Recommendation

Report plain precision alongside mask-aware precision rather than re-tuning the cutoff — the sweep
shows no cutoff in (0, 1) meaningfully changes the story, so a second, unforgiving column is the
only informative addition. The `off` column above is that number. See also
`docs/validation_ap_completeness.md` for the IoU-threshold and incomplete-tile terms of the same gap.
