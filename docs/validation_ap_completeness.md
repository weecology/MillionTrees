# Why AP is far below F1 on the validation split — and what to report

**Question.** Table 5's validation F1 (0.44–0.68) sits far above its AP@50 (0.03–0.31). Two
hypotheses were on the table: (a) the reference data is incompletely annotated, so correct
detections are charged as false positives; (b) AP matches at IoU 0.5 while recall/precision
match at IoU 0.4. Hypothesis (b) was already rejected on the test split
(`ap50_vs_ap40_existing_models.md`: AP40−AP50 ≈ +0.05 against an F1−AP gap of +0.13 to +0.44).

**Answer.** Both matter, but neither is the main driver. The gap is dominated by a third thing:
the F1 in Table 5 is built on **mask-aware precision**, which *forgives* every unmatched
prediction landing on ≥50 % canopy pixels (`MaskAwareDetectionPrecision`,
`tree_fraction_threshold=0.5`). AP has no such forgiveness. On validation imagery that is
~99 % canopy, mask-aware precision forgives almost everything: CanopyRS scores 0.887
mask-aware but **0.508** plain. Roughly 40 % of the gap is that one metric choice.

Dataset **v0.22**, held-out validation split (Allen et al. 2025 + Frey et al. 2026, 85 images).
One inference pass per model, dumped to `outputs/validation_preds/*.pkl` and re-scored offline
by `scripts/rescore_validation_ap.py`. On the `all` regime the re-scorer reproduces the
production evaluator exactly for boxes (e.g. CanopyRS R 0.549 / AP50 0.311 / AP40 0.404) and to
within 0.03 for polygons, so the complete-tile columns are directly comparable to the
`results_*.txt` files the leaderboard is built from. All "overall" rows are macro-averages over
sources, the convention `MillionTreesDataset.eval()` uses.

---

## 1. Is the validation data completely annotated? Partly.

`scripts/validation_completeness.py` audits every validation image and renders a ground-truth
overlay: purple = annotations, yellow dashed = annotated footprint, **red wash = canopy the
independent Restor-TCD segmenter found that nobody annotated** — exactly the pixels where a
correct detection is charged as a false positive. The masks come from a canopy segmenter, never
from the annotations, so this is not circular.

Contact sheets (`docs/public/validation_completeness/`):
`contact_sheet_allen.png`, `contact_sheet_frey_complete.png`, `contact_sheet_frey_edge.png`.
Per-image overlays in `boxes/` and `polygons/`; stats in
`docs/validation_completeness_{boxes,polygons}.csv`.

| Source | Images | Footprint reaches the tile edge | Canopy pixels inside an annotation |
|---|--:|--:|--:|
| Allen et al. 2025 | 24 | 24 / 24 (max margin ≤ 2.4 %) | 0.77 mean (0.56–0.89) |
| Frey et al. 2026 | 61 | 35 / 61 (margin = 0) | **0.99** mean on those 35; 0.16 mean on the other 26 |

Two findings:

- **v0.21 Allen tiles carried a ~12 % unannotated border on all four sides** — only 59 % of each
  tile was inside the annotated footprint. v0.22 re-tiles them to the annotated extent, so the
  border is gone. Anything measured on v0.21 Allen paid a false-positive tax on 41 % of every tile.
- **26 of 61 Frey tiles are edge tiles** of the TLS footprint: the annotations occupy a corner
  and the rest is unlabelled forest. The worst (`ortho_3919_12000_28000`) has 8 annotations in
  the bottom-right corner of a 2000² tile — 99 % of its canopy is unlabelled. The remaining
  **35 interior tiles are annotated wall to wall**. The two Frey contact sheets show this at a
  glance: the interior sheet is almost entirely green, the edge sheet almost entirely red.

Caveat on the Allen column: much of the red on the Spanish (SPA) sites is open shrub and bare
ground that the TCD segmenter calls canopy, so 0.77 is a lower bound on Allen completeness, not
a count of missed trees.

**Complete-tile set used below: 24 Allen + 35 Frey = 59 tiles, 3 661 boxes.**

### Dropping the edge tiles programmatically

The rule is `milliontrees.common.tile_completeness`: a tile is complete when
`max_margin ≤ 0.025` (no side has a strip wider than 2.5 % of the tile outside the annotated
footprint) **and** `canopy_annotated_frac ≥ 0.5` (at least half the segmenter's canopy pixels
fall inside an annotation). Nothing is hand-picked.

`data_prep/flag_tile_completeness.py` applies it to a packaged release and writes
`data_prep/tile_completeness.csv` — one row per tile with both statistics and the verdict, so
the exact tile set behind a published number is reviewable in git:

```bash
python data_prep/flag_tile_completeness.py --version 0.22 --splits validation
#   Allen et al. 2025   24 tiles   24 complete    0 dropped
#   Frey et al. 2026    61 tiles   35 complete   26 dropped
```

All three loaders then take `complete_tiles_only=True`, and every eval script takes
`--complete-tiles-only`:

```python
ds = get_dataset("TreeBoxes", split_scheme="out-of-distribution", complete_tiles_only=True)
len(ds.get_subset("validation"))   # 59, was 85; test split untouched
```

Two details worth knowing:

- **Both tests need annotations with extent**, so the flag is derived once from the box package
  and shared by filename. A stem point sits half a crown inside the tile edge and covers no
  canopy pixels, so scoring TreePoints directly rejects tiles the box version accepts (it kept
  only 4 of 85 in testing). Reading the shared manifest makes all three tasks evaluate the
  **same 59 tiles**, which is what makes their Table 5 rows comparable.
- Rerun the generator after any re-tiling or mask regeneration — tile geometry changes with
  both. If the manifest has no entry for the version in use, the loader computes the rule in
  place rather than silently keeping every tile.

---

## 2. AP50 and AP40, all tiles vs complete tiles

`all` = 85 images (reproduces the reported numbers). `complete` = the 59 wall-to-wall tiles.
R / P are recall and **plain** precision at IoU 0.4 — the same matcher F1 uses, minus the canopy
forgiveness. P (mask) is the mask-aware precision Table 5 reports.

### TreeBoxes

| Model | R | P (mask) | P (plain) | AP50 all | AP40 all | **AP50 complete** | **AP40 complete** |
|---|--:|--:|--:|--:|--:|--:|--:|
| CanopyRS-DINO-SwinL (0.30) | 0.549 | 0.887 | 0.508 | 0.311 | 0.404 | **0.374** | **0.471** |
| SAM3 | 0.538 | 0.613 | 0.278 | 0.251 | 0.340 | 0.312 | 0.404 |
| DeepForest fine-tuned (OOD) | 0.328 | 0.818 | 0.204 | 0.097 | 0.150 | 0.106 | 0.163 |
| DeepForest pretrained | 0.287 | 0.967 | 0.279 | 0.035 | 0.121 | 0.036 | 0.130 |

### TreePolygons

| Model | R | P (mask) | P (plain) | AP50 all | AP40 all | **AP50 complete** | **AP40 complete** |
|---|--:|--:|--:|--:|--:|--:|--:|
| CanopyRS DINO+SAM3 (0.30) | 0.479 | 0.929 | 0.308 | 0.213 | 0.317 | **0.288** | **0.401** |
| Detectree2 | 0.461 | 0.634 | 0.250 | 0.188 | 0.287 | 0.246 | 0.352 |
| SAM3 | 0.477 | 0.604 | 0.246 | 0.184 | 0.270 | 0.248 | 0.341 |
| Mask R-CNN fine-tuned (OOD) | 0.405 | 0.977 | 0.385 | 0.181 | 0.276 | 0.228 | 0.325 |

R / P (mask) / AP50 all / AP40 all are the evaluator's own numbers from the same runs.

### Isolating completeness on the one clean comparison

Frey is the only source with both complete and incomplete tiles, so it isolates the effect with
imagery, model and threshold held fixed — all 61 tiles vs. its 35 interior tiles:

| Model | AP50 all Frey | AP50 interior | P plain all | P plain interior |
|---|--:|--:|--:|--:|
| CanopyRS boxes | 0.310 | **0.437** | 0.536 | 0.724 |
| SAM3 boxes | 0.278 | **0.401** | 0.259 | 0.351 |
| CanopyRS polygons | 0.262 | **0.392** | 0.341 | 0.472 |
| SAM3 polygons | 0.231 | **0.359** | 0.231 | 0.317 |
| Detectree2 | 0.230 | **0.344** | 0.205 | 0.281 |
| Mask R-CNN polygons | 0.182 | 0.275 | 0.275 | 0.379 |
| DeepForest fine-tuned | 0.071 | 0.090 | 0.167 | 0.221 |
| DeepForest pretrained | 0.036 | 0.037 | 0.182 | 0.233 |

Dropping edge tiles lifts AP50 by **+0.09 to +0.13** for every model that localizes well, and
lifts plain precision by a matching amount. It does almost nothing for DeepForest, whose problem
is localization, not the reference data — which is the control that shows the effect is real and
not an artifact of dropping hard tiles.

---

## 3. Decomposition of the F1 → AP50 gap

CanopyRS boxes, the headline model (v0.22, macro over sources). Each row changes exactly one
thing from the row above:

| Step | Value | Change |
|---|--:|--:|
| Table 5 F1 (mask-aware precision 0.887, recall 0.549) | 0.678 | — |
| F1 with **plain** precision (0.508), identical matches | 0.527 | **−0.151** |
| ... as AP40, i.e. ranking-aware, complete tiles | 0.471 | −0.056 |
| ... at IoU 0.5 instead of 0.4, complete tiles | 0.374 | **−0.097** |
| ... on all 85 tiles including edge tiles = reported AP50 | 0.311 | −0.063 |

Of the 0.367 gap: **mask-aware precision ≈ 0.15 (41 %)**, **IoU 0.5 vs 0.4 ≈ 0.10 (26 %)**,
**incomplete edge tiles ≈ 0.06 (17 %)**, and AP's sensitivity to score ranking ≈ 0.06 (15 %),
which F1 at a single operating point ignores entirely.

The per-image detection cap was checked and ruled out: raising torchmetrics'
`max_detection_thresholds` from 100 to 1000 moved box AP by ≤ 0.005.

---

## 4. Recommendation

1. **Report AP40, not AP50, alongside F1.** AP40 uses the same IoU 0.4 that recall and precision
   already use, so a reader comparing the columns is comparing the same notion of a match.
   Publishing AP at an IoU the F1 columns do not use is an unforced inconsistency. AP40 is now a
   standard metric on `TreeBoxes`/`TreePolygons`, computed on the same predictions in the same
   pass — no extra runs needed.
2. **Report validation AP on the complete-tile subset** (24 Allen + 35 Frey), and say so.
   Scoring AP against tiles that are 90 % unlabelled forest measures the annotation protocol,
   not the model.
3. **State plainly that F1's precision is mask-aware.** It is the honest number for "does this
   model find trees" on partially-annotated data, but it is not comparable to AP, and a 0.887
   precision next to a 0.311 AP will invite exactly this question from a reviewer.

Headline validation numbers under that recommendation (complete tiles, AP40):

| | CanopyRS | SAM3 | Detectree2 | MillionTrees fine-tuned |
|---|--:|--:|--:|--:|
| TreeBoxes | **0.471** | 0.404 | — | 0.163 |
| TreePolygons | **0.401** | 0.341 | 0.352 | 0.325 |

---

## Reproducing

```bash
# 1. audit completeness + render GT overlays
uv run python scripts/validation_completeness.py --geometry boxes \
    --out-csv docs/validation_completeness_boxes.csv \
    --viz-dir docs/public/validation_completeness/boxes

# 2. one inference pass per model, dumping raw predictions
sbatch existing_models/slurm/eval_validation_dump_canopyrs.sbatch   # + sam3 / detectree2 / deepforest
sbatch training/slurm/eval_validation_dump_finetuned.sbatch         # fine-tuned boxes
sbatch training/slurm/eval_validation_dump_polygons.sbatch          # fine-tuned polygons

# 3. re-score offline (no GPU): any IoU, any tile subset
uv run python scripts/rescore_validation_ap.py \
    --predictions outputs/validation_preds/*.pkl --max-margin 0.025 \
    --out-csv docs/validation_ap_iou_completeness.csv
```

Jobs: 38129318 (canopyrs) · 38129319 (sam3) · 38129320 (detectree2) · 38129321 (deepforest) ·
38129730 (fine-tuned boxes) · 38129323 (fine-tuned polygons).

Polygon rows differ from the evaluator by ≤ 0.03 on recall/AP (RLE round-trip and empty-mask
handling in the offline matcher); box rows match to three decimals.

### Note on the dataset version

These runs are **v0.22**, not the v0.21 of Table 5. v0.21 can no longer load the validation split
at all: its Allen imagery (3766×3848) no longer matches the regenerated tree-coverage masks
(3007×3090) and the loader raises. v0.22 re-tiles the images to match, which also removes the
unannotated Allen border described in §1. `0.22` has been added to `_versions_dict` in all three
loaders, so it is now the default everywhere — **the rest of the leaderboard must be regenerated
on v0.22 before these numbers go into the same table as test-split numbers**
(`slurm/submit_all.sh` plus the weak-supervision ablation, see `CLAUDE.md` §3).
