# Out-of-distribution test sources, and the `existing_split` leak

Audit of the v0.23 packaged splits (`*_supervised_v0.23/out-of-distribution.csv`),
prompted by the observation that box OOD scores exceed within-distribution scores.
Status: **the OOD split is not clean in any geometry.** A new dataset version is
required after the fix.

## 1. What SelvaBox actually contains

SelvaBox is **not** a compendium and does **not** include BAMFORESTS / Troles et al. 2024.
`data_prep/SelvaBox.py` pulls `CanopyRS/SelvaBox` from HuggingFace; every tile comes from
the project's own 14 tropical drone orthomosaics:

| Ortho prefix | Locality |
|---|---|
| `tbslake` | Tiputini Biodiversity Station, Ecuador |
| `zf2campirana`, `zf2tower`, `zf2quad`, `zf2transectew` | ZF2, Manaus, Brazil |
| `asforestsouth2`, `asforestnorthe2`, `asnortheast`, `asnorthnorth` | Amazon (AS sites) |
| `inundated`, `terrafirme`, `pantano` | forest-type plots |
| `transectotoni`, `sanitower` | Ecuador/Peru transects |

No temperate or German imagery. The BAMFORESTS association comes from **CanopyRS the model**,
not SelvaBox the dataset: CanopyRS trains on Troles Stadtwald and tests on the Hain forest,
which is why `assign_canopyrs_aligned_existing_split` pins Troles rows
(`package_datasets.py:865-869`).

**Related but distinct: SelvaMask.** The polygon source `SelvaMask` is 3 orthos —
`tbsnewsite2` (TBS, Ecuador), `bcifairchildn` (BCI, Panama), `zf2block4` (ZF2, Manaus).
It shares *localities* (TBS, ZF2) with SelvaBox but not orthos or tiles. Since SelvaBox
is moving entirely into train, SelvaMask polygons at the same two localities are a
locality-level overlap to be aware of, especially in the crossgeometry split.

## 2. Declared OOD test sources (`data_prep/package_datasets.py:32-53`)

| Geometry | Declared held-out test sources |
|---|---|
| TreeBoxes | Radogoshi et al. 2021, SelvaBox, NEON_benchmark, Zamboni et al. 2021 |
| TreePoints | Amirkolaee et al. 2023, NEON_points, NEON MultiTemporal, OSBS megaplot 2025, OFO field 2025 |
| TreePolygons | Troles et al. 2024, Bohlman et al. 2008, Lefebvre et al. 2024, NEON MultiTemporal, Takeshige et al. 2025, Alejandro_Miranda |

Train-only override list (`OUT_OF_DISTRIBUTION_TRAIN_ONLY_SOURCES`): OAM-TCD.

## 3. What the OOD test sets actually contain in v0.23

### TreeBoxes — 3,809 images / 316,576 annotations
| Source | Images | Anns | % imgs | Declared? |
|---|---|---|---|---|
| SelvaBox | 1,864 | 199,417 | 48.9 | yes |
| Radogoshi et al. 2021 | 1,533 | 107,173 | 40.2 | yes |
| Zamboni et al. 2021 | 218 | 3,353 | 5.7 | yes |
| NEON_benchmark | 194 | 6,633 | 5.1 | yes |

### TreePoints — 3,260 images / 131,117 annotations
| Source | Images | Anns | % imgs | Declared? |
|---|---|---|---|---|
| OFO field 2025 | 1,387 | 18,513 | 42.5 | yes |
| NEON_points | 1,298 | 50,562 | 39.8 | yes |
| Amirkolaee et al. 2023 | 335 | 53,315 | 10.3 | yes |
| NEON MultiTemporal | 152 | 2,599 | 4.7 | yes |
| OSBS megaplot 2025 | 88 | 6,128 | 2.7 | yes |

### TreePolygons — 2,759 images / 108,473 annotations
| Source | Images | Anns | % imgs | Declared? |
|---|---|---|---|---|
| Lefebvre et al. 2024 | 1,322 | 21,546 | 47.9 | yes |
| **Cloutier et al. 2023** | 425 | 32,377 | 15.4 | **NO** |
| Troles et al. 2024 | 313 | 6,705 | 11.3 | yes |
| **SelvaMask** | 264 | 31,826 | 9.6 | **NO** |
| Takeshige et al. 2025 | 176 | 4,136 | 6.4 | yes |
| NEON MultiTemporal | 151 | 2,590 | 5.5 | yes |
| **NEON combined crowns** | 47 | 2,472 | 1.7 | **NO** |
| **Lucas et al. 2024** | 20 | 1,601 | 0.7 | **NO** |
| Bohlman et al. 2008 | 18 | 3,508 | 0.7 | yes |
| Alejandro_Miranda | 13 | 794 | 0.5 | yes |
| **Zuniga-Gonzalez et al. 2023** | 10 | 918 | 0.4 | **NO** |

**27.8 % of the polygon OOD test set (766 / 2,759 images) comes from sources that are not
declared held-out sources at all.**

## 4. The root cause

Both OOD assignments in `out_of_distribution_split` are gated on `_rows_needing_auto_split`
(`package_datasets.py:887-892`), which skips any row already carrying an `existing_split`
pin of `train`/`test`/`validation`:

```python
TreeBoxes_datasets.loc[
    TreeBoxes_datasets.source.isin(test_sources_boxes)
    & _rows_needing_auto_split(TreeBoxes_datasets),   # <-- pinned rows are exempt
    "split",
] = "test"
```

`apply_existing_splits` runs first and has already placed pinned rows. So the gate leaks in
**both** directions:

- a declared **test** source with `existing_split == "train"` rows keeps those rows in **train**;
- a **train** source with `existing_split == "test"` rows keeps those rows in **test**.

OAM-TCD is the only source with a hard override
(`df.loc[df.source.isin(OUT_OF_DISTRIBUTION_TRAIN_ONLY_SOURCES), "split"] = "train"`).
Nothing gives the declared test sources the equivalent.

The comment at `package_datasets.py:1289-1290` — "drop excess images instead of demoting to
train so held-out sources never leak into the train split" — describes an invariant that
`limit_test_images` alone cannot enforce, because the leak happens upstream of it.

## 5. Same-source train/test leaks in v0.23

| Geometry | Source | Train imgs | Train anns | Test imgs | Test anns |
|---|---|---|---|---|---|
| TreeBoxes | SelvaBox | 585 | 231,932 | 1,864 | 199,417 |
| TreePoints | Amirkolaee et al. 2023 | 277 | 44,326 | 335 | 53,315 |
| TreePoints | OFO field 2025 | 6,413 | 117,741 | 1,387 | 18,513 |
| TreePolygons | Troles et al. 2024 | 2,140 | 85,562 | 313 | 6,705 |

Radogoshi, Zamboni, NEON_benchmark, NEON_points, NEON MultiTemporal, OSBS megaplot,
Bohlman, Lefebvre, Takeshige and Alejandro_Miranda are clean (0 train rows) — they carry
no upstream `existing_split == "train"` pin.

## 6. Consequence for the published leaderboard

The box OOD aggregate exceeds the within-distribution aggregate not because OOD is easier
per image, but because the headline "Average" is an unweighted macro-mean over sources
(`milliontrees_dataset.py:546-560`) and the two splits average over different source sets —
16 sources vs 4. On the four sources present in both test sets the fine-tuned model is
**worse** OOD (recall 0.709 → 0.644, AP40 0.591 → 0.480). Every never-trained model shows
the same aggregate rise with flat-or-worse per-source numbers, which is proof the effect is
test-set composition, not the model.

## 7. Fix

1. Remove `SelvaBox` from `OUT_OF_DISTRIBUTION_TEST_SOURCES_BOXES` and add it to
   `OUT_OF_DISTRIBUTION_TRAIN_ONLY_SOURCES` (decision: keep the 231k tropical boxes as
   training data rather than salvage a broken hold-out).
2. Give declared OOD test sources a hard override symmetric to the train-only one, so
   `existing_split` pins cannot override OOD membership in either direction.
3. Re-package and bump `_versions_dict` in all three loaders.
4. Re-run all three leaderboard components (see `CLAUDE.md` §3).

Watch out: `limit_test_images` short-circuits and returns `df` unchanged whenever *any* row
carries `existing_split == "test"` (`package_datasets.py:1179-1180`) — that check is global,
not per source. Fixing the pins must not silently re-enable the 50-image-per-source cap on
the OOD test sets.

## 8. Further leaks found (cross-geometry and same-image)

### 8a. Crossgeometry: 151 polygon test images are also in box AND point train

`cross_geometry_split` puts **all** boxes and **all** points into train
(`package_datasets.py:1133-1136`) and filters polygons to the declared OOD polygon test
sources. NEON MultiTemporal is both a polygon OOD test source and a box/point train source,
so the identical tiles appear on both sides:

| Polygon test images also in crossgeometry train | Count | Source |
|---|---|---|
| ...in TreeBoxes train | 151 of 4,133 | NEON MultiTemporal |
| ...in TreePoints train | 151 of 4,133 | NEON MultiTemporal |

These are exact filename matches, not locality overlap. The crossgeometry task is
"predict polygons from another geometry with no local data from the test localities in
train" — 151 tiles violate that directly.

### 8b. Same filename in both train and test within one split scheme

| Split scheme | Geometry | Filenames in both train and test |
|---|---|---|
| within-distribution | TreeBoxes | 13 |
| within-distribution | TreePoints | 7 |
| within-distribution | TreePolygons | 22 |
| out-of-distribution | TreePolygons | 5 |

Small, but a single image should never be on both sides. Likely rows of one image
disagreeing on `existing_split`, or a filename collision from `assign_packaged_filenames`.

### 8c. The undeclared polygon test sources are also same-source train+test

| Source | Test imgs | Train imgs |
|---|---|---|
| Cloutier et al. 2023 | 425 | 1,948 |
| SelvaMask | 264 | 235 |
| Lucas et al. 2024 | 20 | 8 |
| Zuniga-Gonzalez et al. 2023 | 10 | 27 |
| NEON combined crowns | 47 | 0 |

TreeBoxes and TreePoints have no undeclared sources in their OOD test sets.

## 9. Projected composition after the symmetric fix

Declared OOD test sources forced entirely to test, everything else entirely to train,
SelvaBox moved to train-only. Cap disabled.

| Geometry | Current test | Proposed test | Proposed train |
|---|---|---|---|
| TreeBoxes | 3,809 imgs / 4 sources | **1,945 imgs / 3 sources** | 14,616 imgs / 13 sources |
| TreePoints | 3,260 imgs / 5 sources | **9,950 imgs / 5 sources** | 25,303 imgs / 4 sources |
| TreePolygons | 2,759 imgs / 11 sources | **4,133 imgs / 6 sources** | 6,911 imgs / 19 sources |

Per-source shares after the fix:

- **TreeBoxes**: Radogoshi 78.8 %, Zamboni 11.2 %, NEON_benchmark 10.0 %. Losing SelvaBox
  leaves the box OOD test set dominated by a single Nordic source.
- **TreePoints**: OFO field 2025 78.4 %, NEON_points 13.0 %, Amirkolaee 6.2 %,
  NEON MultiTemporal 1.5 %, OSBS megaplot 0.9 %. Test set triples; train drops to 4 sources.
- **TreePolygons**: Troles 59.4 %, Lefebvre 32.0 %, Takeshige 4.3 %, NEON MultiTemporal 3.7 %,
  Bohlman 0.4 %, Alejandro_Miranda 0.3 %.

Note the macro-average over sources (§6) means these share percentages do not directly drive
the headline number, but the *number of sources* does: boxes would drop from 4 to 3.

## 10. Point OOD test composition — two problems

### 10a. Amirkolaee is the pretrained checkpoint's own training set

`weecology/deepforest-tree-point` — the "TreeFormer (release weights)" leaderboard row — is
trained on the Amirkolaee (KCL) source (`notes/treeformer_points_diagnosis_plan.md:164-168`,
flagged there as **E8** and never resolved). Amirkolaee et al. 2023 *is* the TreeFormer paper
(`docs/datasets.md:240`, London, England). So Amirkolaee cannot be an out-of-distribution
test source for that row.

**The empirical direction is the opposite of what you would assume.** On the published v0.23
OOD run, Amirkolaee is the pretrained model's *worst* source, not an inflated one:

| Source | Pretrained | Fine-tuned |
|---|---|---|
| Amirkolaee et al. 2023 | **0.291** | 0.623 |
| NEON MultiTemporal | 0.981 | 0.781 |
| NEON_points | 0.951 | 0.597 |
| OFO field 2025 | 0.951 | 0.976 |
| OSBS megaplot 2025 | 0.641 | 0.465 |
| **macro (published)** | **0.763** | **0.688** |
| **macro without Amirkolaee** | **0.881** | **0.705** |

Removing Amirkolaee *widens* the pretrained-over-fine-tuned gap from +0.075 to +0.176. It
should still come out — evaluating a model on its own training data is invalid regardless of
which way the bias runs — but doing so strengthens the leaderboard's "TreeFormer fine-tunes
well and generalizes poorly" claim rather than undermining it.

Likely cause of the low score on its own training data: the GSD-normalized match radius for
Amirkolaee is 0.0195 (4 m at London's resolution), far stricter in pixels than the 0.10 used
for the NEON sources, compounded by the `density_sigma=3.0` merged-peak problem
(`notes/treeformer_hyperparameters.md:54`).

### 10b. The point OOD test set is three-quarters NEON, with site-level pseudo-replication

Site tokens extracted from packaged filenames:

| Source | Sites |
|---|---|
| NEON_points | HARV, SERC, OSBS, BLAN, NIWO, RMNP, BONA, WREF, UNDE, BART, DEJU, TREE, ABBY |
| NEON MultiTemporal | HARV, BART, TEAK, MLBS, WREF, ABBY |
| OSBS megaplot 2025 | OSBS |

NEON MultiTemporal shares HARV, BART, WREF and ABBY with NEON_points; OSBS megaplot 2025 is
entirely at OSBS, which NEON_points also covers (2,121 annotations). These are different
imagery and years, not duplicate tiles, so this is not a train/test leak — but because the
headline is an unweighted macro-mean **over sources**, the NEON network carries 3 of 5 votes
(4 of 5 once Amirkolaee is removed) while being one sampling program at overlapping sites.

### 10c. Full point source inventory

| Source | Images | Anns | Location | Current role |
|---|---|---|---|---|
| Beery et al. 2022 | 22,437 | 451,007 | — | **hard train-only** (`TRAIN_ONLY_SOURCES`) |
| OFO field 2025 | 7,800 | 136,254 | Open Forest Observatory | OOD test (leaking) |
| Ventura et al. 2022 | 1,645 | 96,509 | Southern California urban, USA | train |
| NEON_points | 1,298 | 50,562 | NEON network | OOD test |
| Chen & Shang (2022) | 1,146 | 98,949 | Yosemite NP, California, USA | train |
| Amirkolaee et al. 2023 | 612 | 97,641 | London, England | OOD test (leaking) |
| NEON MultiTemporal | 152 | 2,599 | NEON network | OOD test |
| OSBS megaplot 2025 | 88 | 6,128 | OSBS, Florida, USA | OOD test |
| Dubrovin et al. 2024 | 75 | 6,925 | Perm Krai, Russia | train |

Only 9 point sources exist, so every hold-out decision is expensive: holding out 6 leaves 3
for training.

## 11. Decisions taken and the resulting splits

Implemented in `data_prep/package_datasets.py` (not yet repackaged).

| Source | Geometry | Decision |
|---|---|---|
| SelvaBox | boxes | → `OUT_OF_DISTRIBUTION_TRAIN_ONLY_SOURCES` (all 2,449 imgs / 431k boxes to train) |
| NEON MultiTemporal | boxes | → added to box hold-out (also already the point and polygon hold-out) |
| Amirkolaee et al. 2023 | points | → removed from hold-out (it is the pretrained checkpoint's training set) |
| Dubrovin et al. 2024 | points | → added to hold-out (Perm Krai, Russia — the non-NEON vote) |
| OFO field 2025 | points | → genuine hold-out (6,413 train imgs moved to test) |
| Troles et al. 2024 | polygons | → genuine hold-out (2,140 train imgs moved to test) |
| Cloutier, SelvaMask, NEON combined crowns, Lucas, Zuniga-Gonzalez | polygons | → train (were leaking into test) |

Code changes:
1. `_assign_ood_split_by_source` replaces the `_rows_needing_auto_split`-gated assignment in
   `out_of_distribution_split`. OOD membership is now decided by source alone; validation
   rows are protected.
2. `cross_geometry_split` now drops box/point rows whose source is a declared polygon
   hold-out before the blanket train assignment. Promoting NEON MultiTemporal to the box
   hold-out does **not** fix this on its own — crossgeometry assigns all boxes and points to
   train unconditionally, independent of OOD membership.
3. Constants updated per the table above.

### Simulated result on v0.23 source data

| Geometry | Train | Test | Test sources |
|---|---|---|---|
| TreeBoxes | 14,464 imgs / 902,412 anns / 12 src | 2,097 imgs / 119,761 anns | Radogoshi 1,533 · Zamboni 218 · NEON_benchmark 194 · NEON MultiTemporal 152 |
| TreePoints | 25,840 imgs / 744,106 anns / 4 src | 9,413 imgs / 202,468 anns | OFO 7,800 · NEON_points 1,298 · NEON MT 152 · OSBS megaplot 88 · Dubrovin 75 |
| TreePolygons | 6,911 imgs / 327,392 anns / 19 src | 4,133 imgs / 124,841 anns | Troles 2,453 · Lefebvre 1,322 · Takeshige 176 · NEON MT 151 · Bohlman 18 · Alejandro_Miranda 13 |

Verified: every declared hold-out has 0 train rows; train/test filename overlap is 0 in all
three geometries; crossgeometry box/point train no longer shares any filename with polygon
test (152 NEON MultiTemporal images dropped from each). `limit_test_images` still
short-circuits on the `existing_split == "test"` check, so the 50-image cap stays off as before.

### Still open

- The macro-average still gives NEON 3 of 5 point votes (NEON_points, NEON MultiTemporal,
  OSBS megaplot at overlapping sites). Ventura et al. 2022 (SoCal urban) remains the
  obvious further hold-out if that weighting becomes a problem.
- The within-distribution same-filename train/test overlaps (13 boxes / 7 points / 22
  polygons, §8b) are untouched — separate cause, not the OOD gate.
- Repackage + `_versions_dict` bump in all three loaders, then rerun all three leaderboard
  components (`CLAUDE.md` §3).

## 12. Beery et al. 2022 (AutoArborist) relabelled as unsupervised

AutoArborist labels are municipal street-tree **inventory records** projected into aerial
imagery, not image annotations. Evidence that this is weak supervision rather than
supervision:

| | Rows |
|---|---|
| Raw AutoArborist | 898,550 |
| TCD canopy-filtered (what ships, `annotation_csvs.cfg:40`) | 452,367 |
| **Discarded for landing on majority no-tree pixels** | **446,183 (49.7 %)** |

Surviving that filter only means a point hit *some* canopy — not the correct tree, not the
crown centre. The source is also non-exhaustive by construction (city inventories cover
street and public trees; the tiles also contain park, yard and private trees), which is why
it already carries `complete = False`.

The project already refuses to evaluate against it — `TRAIN_ONLY_SOURCES` in both
`package_datasets.py` and `TreePointsDataset` — yet in the supervised release it supplied
**86.8 % of point train images** (22,437 / 25,840) and **60.6 % of point train annotations**
(451,007 / 744,106). A source too unreliable to score against should not dominate what the
point models learn from.

Change: `"Beery et al. 2022"` added to `UNSUPERVISED_SOURCE_ALIASES` and
`"Beery et al. 2022 unsupervised"` to `UNSUPERVISED_SOURCES_EXPECTED["TreePoints"]`.
`normalize_unsupervised_sources` renames it at packaging time, the loaders' default
`*unsupervised*` exclusion drops it, and it stays reachable via `include_unsupervised=True`
— i.e. it moves into the weak-supervision pool the pretraining ablation (CLAUDE.md §3) exists
to exploit. Verified idempotent (re-running on an already-suffixed frame is a no-op).

`TRAIN_ONLY_SOURCES` keeps the unsuffixed entry: v0.23 and earlier shipped that name, and the
loader-side demotion still has to handle released CSVs.

Resulting supervised point train:

| | Before | After |
|---|---|---|
| Images | 25,840 | 3,403 |
| Annotations | 744,106 | 293,099 |
| Sources | 4 | 3 (Ventura, Chen & Shang, Amirkolaee) |

Expect the fine-tuned point leaderboard row to move.
