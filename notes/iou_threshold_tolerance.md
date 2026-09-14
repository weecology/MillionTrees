# What AP40 accepts that AP60 rejects

Supplementary material for the AP reporting scheme (`CLAUDE.md` §4). MillionTrees reports
**AP40** as the headline — predictions matched to ground truth at IoU 0.4, the same IoU
behind the recall and mask-aware precision that make up F1, so AP and F1 agree on what
counts as a match — and **AP60** alongside it as the strict complement. AP60 is never
reported alone; its job is the AP40 → AP60 *drop*.

This note gives that drop a physical meaning: the geometry a prediction may get wrong and
still be counted a match at IoU 0.4 but not at IoU 0.6.

![IoU tolerance figure](../docs/public/iou_tolerance_figure.svg)

Regenerate with:

```bash
python scripts/make_iou_tolerance_figure.py \
    --out docs/public/iou_tolerance_figure.png --also-svg \
    --table-out notes/iou_threshold_tolerance_table.md
```

No GPU, no data, no model — it is closed-form geometry. `--loose` / `--strict` take any
pair, e.g. `--loose 0.4 --strict 0.5` to compare against the conventional threshold.

## The quantities

All follow from IoU = I / (A + B − I), with A = ground-truth area, B = prediction area,
I = intersection. Each isolates one error mode by holding the others fixed.

| Quantity | Error mode it measures | Formula (L = crown width) | IoU 0.4 | IoU 0.6 | ratio |
|---|---|---|---|---|---|
| **Nested prediction width** | correctly centred but too small, entirely inside the GT | √IoU · L | 0.632 L | 0.775 L | — |
| **Uncovered margin per side** | strip of crown that nested prediction fails to cover | (1 − √IoU)/2 · L | **0.184 L** | **0.113 L** | 1.63× |
| **Max centroid offset** | right size, mislocated along one axis | (1 − IoU)/(1 + IoU) · L | **0.429 L** | **0.250 L** | 1.71× |
| **Max disagreement area** | any error mode — missed plus over-predicted area | (A + B)(1 − IoU)/(1 + IoU) | 0.857 A ᵃ | 0.500 A ᵃ | 1.71× |
| **Permitted size range** | prediction area relative to GT, best case | IoU ≤ min(r, 1/r) | 0.40–2.50 A | 0.60–1.67 A | — |

ᵃ for a prediction the same size as the GT (B = A). The disagreement-area identity holds
for **any** two shapes at any offset — an IoU threshold is exactly a cap on symmetric
difference, scaled by the summed areas. Verified against 2×10⁵ random box pairs to 7e-15.
Going from 0.6 to 0.4 raises that cap by (5/28)(A + B) ≈ 0.36 A.

Every row is **scale-free**: a ratio of L, not a number of metres or pixels. IoU cancels
resolution entirely. In pixel terms, for a crown 100 px across: **43 px vs 25 px** of
permitted centroid offset, **18 px vs 11 px** of permitted shrink, at every GSD.

## A crown 100 px across, at each GSD

| GSD | Crown width | Crown area | Margin/side 0.4 | 0.6 | gain | Max offset 0.4 | 0.6 | gain | Extra disagreement area |
|---|---|---|---|---|---|---|---|---|---|
| 1 cm | 1.00 m | 1 m² | 0.18 m | 0.11 m | +0.07 m | 0.43 m | 0.25 m | +0.18 m | +0.4 m² |
| 3 cm | 3.00 m | 9 m² | 0.55 m | 0.34 m | +0.21 m | 1.29 m | 0.75 m | +0.54 m | +3.2 m² |
| 5 cm | 5.00 m | 25 m² | 0.92 m | 0.56 m | +0.36 m | 2.14 m | 1.25 m | +0.89 m | +8.9 m² |
| 10 cm | 10.00 m | 100 m² | 1.84 m | 1.13 m | +0.71 m | 4.29 m | 2.50 m | +1.79 m | +35.7 m² |
| 30 cm | 30.00 m | 900 m² | 5.51 m | 3.38 m | +2.13 m | 12.86 m | 7.50 m | +5.36 m | +321.4 m² |
| 50 cm | 50.00 m | 2500 m² | 9.19 m | 5.64 m | +3.55 m | 21.43 m | 12.50 m | +8.93 m | +892.9 m² |

**Column by column:**

- **GSD** — ground sample distance, metres per pixel. The only input that varies.
- **Crown width** — 100 px × GSD. The physical side length of the square crown box. *(At
  100 px per crown this equals the GSD in cm: 5 cm/px → a 5 m crown. That coincidence is
  why the figure needs only one x-axis.)*
- **Crown area** — width², the ground-truth box area A. Grows as the square, which is why
  the last column explodes while the linear columns do not.
- **Margin/side 0.4, 0.6, gain** — the strip of crown a correctly-centred prediction may
  leave uncovered on each edge and still match. The `gain` column is the slack AP40 has
  and AP60 does not, per edge.
- **Max offset 0.4, 0.6, gain** — how far a correctly-sized prediction's centre may sit
  from the true centre and still match, measured along one axis.
- **Extra disagreement area** — the additional missed-plus-over-predicted area IoU 0.4
  tolerates over 0.6, for a same-size prediction: (5/28)(A + B) ≈ 0.36 A.

## How to read this

**The gap is large and roughly constant in relative terms.** IoU 0.4 permits ~63 % more
uncovered crown per edge than IoU 0.6, and ~71 % more centroid displacement — at every
resolution and every crown size. That is a much wider separation than 0.4-vs-0.5 (26 % and
29 % respectively), which is why the pair is informative: AP40 and AP60 are not two views
of the same tolerance, they are qualitatively different questions.

**What the AP40 → AP60 drop measures.** At 10 cm the modal crown is ~10 m. AP40 accepts a
prediction whose centre is up to 4.29 m off, or which covers only the middle 6.3 m of the
crown; AP60 requires the centre within 2.50 m and coverage of the middle 7.7 m. So AP40
mostly asks *did the model find this tree*, and AP60 asks *did it draw the crown*. A model
that finds trees but delineates them loosely scores well on AP40 and collapses at AP60 —
which is exactly the pattern in `notes/leaderboard_validation.md`: on the validation split
boxes retain 6–53 % of their AP40 at AP60 and spread widely, while polygons all sit at
33–37 % and separate nobody.

**The table sweeps crown size, not resolution.** Holding a crown at 100 px while varying
GSD varies the physical tree — a 1 m sapling at 1 cm/px, an unrealistic 50 m crown at
50 cm/px. Read a row as *"if a crown occupies ~100 px in my tiles, here is what the
threshold costs on the ground"*, which is the frame a detector actually operates in. Do
not read the 30 cm and 50 cm rows as claims about coarse imagery; they are claims about
very large crowns.

**Why the headline is 0.4 and not 0.5 or 0.6.** At 10 cm, IoU 0.6 permits only 1.13 m of
uncovered crown per edge on a 10 m tree. Human delineation disagreement on a crown that
size — crown overlap, shadow, leaning boles — is of comparable magnitude, so a threshold
that strict is partly scoring annotator noise rather than model error. IoU 0.4 (1.84 m per
edge) sits outside that band. AP60 is retained precisely because it sits *inside* it: as a
deliberate stress test of delineation, read as a ratio against AP40, never on its own.

**One regime where the distinction is arbitrary.** For small crowns at coarse GSD — a 2 m
crown at 50 cm/px is 4 px across — the two thresholds sit at 1.7 px vs 1.0 px of permitted
offset. There the AP40/AP60 gap is decided by sub-pixel rounding, not by anything
meaningful about delineation quality.
