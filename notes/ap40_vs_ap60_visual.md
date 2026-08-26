# What the AP40 → AP60 drop looks like (fine-tuned DeepForest, validation)

The fine-tuned box model scores **AP40 0.164 / AP60 0.053** on the held-out validation split —
it keeps 32 % of its AP when the match threshold moves from IoU 0.4 to 0.6. This note shows the
boxes that account for the difference, so the number has a picture attached.

Built by `scripts/visualize_ap40_vs_ap60.py` from
`outputs/validation_preds/deepforest_finetuned_boxes_validation_v0.23.pkl`, a `--save-predictions`
dump of the exact checkpoint behind the published row (`boxes-best.ckpt`, job 40085567, which
reproduced AP40 0.164 / AP60 0.053 / R 0.328 / P 0.880). Every prediction above the 0.1 eval
threshold is bucketed by the repo's own matcher (`greedy_iou_match`) run twice per image.

## The buckets

| | count | note |
|---|--:|---|
| Predictions above threshold | 5 980 | |
| Matched at IoU 0.4 | 1 108 | what AP40 can score |
| — of which survive IoU 0.6 | 531 | what AP60 can score |
| — of which **lost between 0.4 and 0.6** | **577** | the subject of this note |
| Matched at neither | 4 872 | false positives at both IoUs |

**52 % of everything AP40 counts as a hit is a box that AP60 rejects**, and the split is
identical on both sources (Allen 343/659, Frey 234/449) — this is a property of the model, not
of one site.

## The failure mode is under-drawing, not mis-placement

| bucket | median area(pred)/area(GT) | under-drawn (<0.8) |
|---|--:|--:|
| lost at IoU 0.6 | **0.57** | 84 % |
| survives IoU 0.6 | 0.77 | 58 % |

The lost boxes are not correctly-sized boxes in the wrong place — they are boxes covering
roughly **half the crown's area**, usually centred on the bright sunlit core of the crown with
the shaded outer branches left outside. That is visible in every close-up: the orange prediction
sits inside the purple ground-truth crown rather than beside it. A model that mis-placed boxes
would show area ratios near 1.0 and the two buckets would look alike.

This is why AP60 is worth reporting next to AP40 rather than instead of it: AP40 alone says the
model finds ~1 100 crowns, and cannot say that it draws half of them at roughly half size.

## Every Allen box, at full size

All **343** Allen et al. 2025 boxes that AP40 counts and AP60 rejects are rendered at
`../docs/public/ap40_vs_ap60/allen/`, 24 per sheet over 15 sheets, plus
`ap40_only_closeups_allen_all.pdf` with every sheet in one file:

```bash
python scripts/visualize_ap40_vs_ap60.py \
  --predictions outputs/validation_preds/deepforest_finetuned_boxes_validation_v0.23.pkl \
  --image-dir /orange/ewhite/web/public/MillionTrees/TreeBoxes_v0.23/images \
  --out-dir docs/public/ap40_vs_ap60/allen \
  --source "Allen" --all-crops --crops-per-sheet 24
```

**The sheets are ordered by IoU, so they read as a progression** and each one is titled with its
band. That ordering is the point of looking at all 343 rather than a sample:

| sheets | IoU band | n | area(pred)/area(GT) | what the boxes look like |
|---|---|--:|---|---|
| 1–3 | 0.40–0.46 | 90 | median 0.53 (88 % under 0.8) | Unambiguous under-drawing: the box sits on the sunlit centre of the crown with the whole shaded margin outside it. Nobody would call these correct. |
| 4–12 | 0.46–0.56 | 171 | median 0.63 (75 % under 0.8) | The transition. Boxes cover most of the crown but stop short on one or two sides. |
| 13–15 | 0.56–0.61 | 82 | median 0.65 (73 % under 0.8) | **Boxes most reviewers would accept by eye** — several are near-perfect crown boxes that miss the 0.6 bar by a hundredth. |

Two things to take from that table. First, the last group is the honest caveat on AP60: at the
top of the band it rejects predictions that are, visually, right. AP60 is a useful *relative*
measure — the 0.05-to-0.58 retention spread across box models is real and large — but it should
not be read as "only 32 % of this model's detections are good crowns". Sheets 13–15 are why the
AP40 → AP60 *drop* is the reportable quantity rather than AP60 on its own.

Second, **the size error barely improves across the band** (median area ratio 0.53 → 0.63 → 0.65;
IoU-vs-area-ratio correlation just 0.13). A box at IoU 0.59 is not meaningfully better *sized*
than one at IoU 0.41 — it is better *centred*. The model under-draws crowns at roughly a constant
~0.6× area regardless of how well it localises them, so the AP40 → AP60 drop is measuring a
systematic scale bias, not a long tail of sloppy outliers. That is a fixable calibration problem
(box-regression targets, anchor scales, or a fixed dilation at inference), which the AP40 number
alone gives no hint of.

Per-source split, for reference: Allen loses 343 of 659 AP40 matches (52 %), Frey 234 of 449
(52 %) — the same rate, so the Allen sheets are representative of the model's behaviour, not of
one site's annotation style.

## Figures

- `../docs/public/ap40_vs_ap60/matched_iou_histogram.png` — IoU of every AP40-matched prediction,
  with the AP60 cutoff drawn on it. The mass is spread fairly evenly across 0.4–0.6 rather than
  piled at either edge, so AP60 is not a threshold that happens to clip an outlier tail; moving
  the bar anywhere in that range removes a comparable slice.
- `../docs/public/ap40_vs_ap60/ap40_only_closeups.png` — 12 individual lost boxes sampled evenly
  across the 0.4–0.6 band (purple dashed = GT crown, orange = prediction). Sampling the
  *lowest*-IoU ones instead returns a page of near-identical IoU 0.40 boxes, the matcher's floor.
- `../docs/public/ap40_vs_ap60/contact_sheet.png` — the six tiles with the most lost boxes, whole
  frame: green survives IoU 0.6, orange is counted by AP40 and rejected by AP60.
- `../docs/public/ap40_vs_ap60/bucket_counts.csv` — per-image bucket counts.

## Reproducing

```bash
python scripts/visualize_ap40_vs_ap60.py \
  --predictions outputs/validation_preds/deepforest_finetuned_boxes_validation_v0.23.pkl \
  --image-dir /orange/ewhite/web/public/MillionTrees/TreeBoxes_v0.23/images \
  --out-dir docs/public/ap40_vs_ap60
```

The script is geometry-agnostic for boxes and takes `--iou-lo` / `--iou-hi`, so the same view can
be built for any other box model with a dump (CanopyRS, SAM3, pretrained DeepForest).
