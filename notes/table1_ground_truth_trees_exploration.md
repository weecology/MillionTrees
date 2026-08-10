# Table 1 — Ground-truth tree counts: a source-by-source exploration

**Purpose.** Manuscript Table 1 reports *annotation counts* (number of geometry rows
per source). That is **not** the number of trees on the ground. Two mechanisms inflate
the annotation count above the true number of physical trees:

1. **Multi-scan / multi-temporal re-annotation.** A single geographic area is imaged by
   several airborne/drone acquisitions, and the same crown map (or field inventory) is
   projected onto *every* acquisition. Each physical tree is then counted once per scan.
2. **Tiling fragmentation.** A source raster is split into tiles; a crown that straddles a
   tile boundary is clipped into two (or more) partial geometries, each counted separately.

This document estimates a **fourth column** for Table 1 — *estimated ground-truth trees* —
source by source, states the method, and flags the sources that need a human to open the
upstream files before we trust a number. It is a **starting point**, not a finished audit.

> **Caveat up front:** the numbers in the current manuscript Table 1 are stale relative to
> the packaged CSVs. Several sources have changed materially (e.g. SelvaBox table=131,443
> vs current CSV=431,910; Cloutier table=259,304 vs current=160,021; Radogoshi table=115,683
> vs current=101,837). The "annotation count" column below is recomputed from the current
> upstream CSVs in `data_prep/annotation_csvs.cfg`.

---

## How tiling is handled (why fragmentation is usually *not* the main problem)

Most polygon sources are tiled through `data_prep/preprocess_polygons.py::select_annotations`
with `patch_overlap=0`. That function clips each polygon to the tile window and then
**keeps a clipped piece only if it retains >50 % of the original crown area**
(`preprocess_polygons.py:83`). Consequently a crown split across a boundary contributes at
most one geometry, not two — tiling fragmentation is largely self-correcting for these
sources. Points are kept whole. The dominant inflation is therefore **multi-scan
re-annotation**, not tiling, for the polygon sources.

Box sources tiled through DeepForest's own `split_raster` do not all share this >50 % rule,
so box tiling still deserves a per-source look (flagged below).

---

## Category A — Multi-scan / multi-temporal (annotation count ≫ trees). HIGH PRIORITY

| Source | Geom | Annotations (current CSV) | Est. ground-truth trees | Inflation | Evidence |
|---|---|---:|---:|---:|---|
| **Cloutier et al. 2023** | Polygons | 160,021 | **~22,860** | ~7× | 7 flight dates × 3 zones (z1/z2/z3) of one forest (SBL). Per-zone counts are near-identical across dates (z1 ≈ 10,665 on every date), i.e. the same crown map re-projected onto each acquisition. GT ≈ one date: z1 10,665 + z2 7,562 + z3 4,638. |
| **NEON MultiTemporal** | Boxes/Points/Polys | 2,602 / 2,599 / 2,590 | **784** | ~3.3× | `individual` column: 784 unique individuals (points 783, polys 783). Same trees imaged across years. |
| **BCI / Vasquez (BohlmanBCI + BCI crownmaps)** | Polygons | 2,440 (2020) + 2,725 (2022) = 5,165 | **~2,700** | ~1.9× | Two crown maps (2020-08-01, 2022-09-29) of the **same 50 ha ForestGEO plot**. Trees counted once per campaign. Union ≈ one campaign's crown count. Needs geometric matching to confirm. |
| **Open Forest Observatory 2026 (field)** | Points | 136,372 | **~30,000–45,000 (uncertain)** | ~3–4× | 237 "missions" but the same field plot is paired to multiple drone missions: missions 934 & 935 both have *exactly* 10,592 rows; 7 missions share exactly 4,250; 117,263 of 136,372 rows (86 %) live in repeated-size mission groups. Field stems re-paired per flight. **Needs field-inventory-level dedup.** |

**Notes.**
- Cloutier and NEON MultiTemporal are **out-of-distribution test** sources; the inflated
  count does not affect training totals but *does* misrepresent Table 1.
- The Cloutier estimate still contains within-date tiling, but the >50 % rule keeps that small.
- BCI: BohlmanBCI (`annotations_crowns.csv`) is a *separate* older Bohlman crown map (see
  Category B); the 2020/2022 `BCI_50ha_*_crownmap` files are the ForestGEO maps. Confirm
  whether all three describe overlapping stems before summing.

---

## Category B — Tiled master crown map, dedup available via crown ID. MEDIUM PRIORITY

These carry a stable per-crown ID, so the true tree count is directly recoverable.

| Source | Geom | Annotations | Est. ground-truth trees | Method |
|---|---|---:|---:|---|
| **HARV Field / Johnson et al. 2021** | Polygons | 9,252 | **1,333** | `crownID` nunique = 1,333 (GlobalID = 1,757, StemTag = 1,084). 47 tiles from 7 rasters ⇒ ~7× tiling inflation. |
| **BohlmanBCI (Bohlman et al. 2008)** | Polygons | 3,514 | **~1,815** | `UNIQUE` nunique = 1,815 / `NEWNO` = 1,814 across 18 tiles. Crown map tiled ⇒ ~1.9× inflation. |

These two can be corrected *now* with a groupby on the ID column — no human needed, just a
decision to dedup at packaging time.

---

## SelvaBox resolved — 431,910 tiled instances ≈ 83,000 unique crowns

Audited directly from the 64 canonical HF parquet shards
(`/orange/ewhite/DeepForest/SelvaBox/cache/data`), which carry provenance columns our
packaged CSV drops: `raster_name`, `location`, `fold`, and a `tile_metadata` struct with
CRS + affine transform + bounds.

**Not a compendium.** All 431,910 boxes come from 14 SelvaBox-native drone orthomosaics at
three tropical sites — brazil_zf2 (4 rasters, 93,931), ecuador_tiputini (6, 150,975),
panama_aguasalud (4, 187,004). No annotations from Troles, OAM-TCD, or any other
MillionTrees source are mixed in. SelvaMask's 3 rasters (`20240131_zf2block4_ms`,
`20240613_tbsnewsite2_m3e`, `20241122_bcifairchildn_m3m`) are **disjoint** from these 14,
so its 70,417 polygons are not the same trees in a second geometry.

**The inflation is tile overlap, and it is large.** Train tiles are 3555 px at 50 % overlap;
valid 1777 px at 50 %; test 1777 px at **75 %** — so a test crown can appear in up to 16
tiles. That is why test carries 161,188 boxes from ~13 % of the area while train carries
232,071 from ~74 %.

Projecting every box into its raster's CRS via the tile affine transform and union-find
merging at geographic IoU ≥ 0.5 (`scripts/`-style one-off, see git history of this doc):

| | tiled boxes | unique crowns | dup factor |
|---|---:|---:|---:|
| Total | 431,910 | 84,548 | 5.11× |
| minus clusters made only of edge-clipped boxes | | 81,148 | |
| per fold: train | 232,071 | 62,628 | 3.71× |
| per fold: valid | 38,651 | 10,108 | 3.82× |
| per fold: test | 161,188 | 11,834 | 13.62× |

11.9 % of boxes touch a tile border, and a crown clipped at a boundary does not merge with
its counterpart — so 84,548 is a slight *over*count and 81,148 a slight undercount. The
authors' **83,137** ("over 83 000 unique human bounding box annotations", arXiv 2507.00170)
falls between the two, independently confirming both their figure and our full download.

**Three numbers, three meanings:** 83,137 = unique crowns on the ground (use for Table 1
"trees"); 431,910 = tiled instances the model sees (use for train/eval volume);
**131,443 = neither** — a stale artifact of the pre-2026-06-17 partial download that pulled
HuggingFace's auto-converted `"partial":true` export (7 of 14 orthos). Remove it wherever
it still appears.

**Gap to close:** `data_prep/SelvaBox.py` writes only `source='SelvaBox'` and discards
`raster_name`, `location`, and the affine transform. Carrying those three fields into
`annotations.csv` costs nothing and makes per-site reporting and on-demand dedup trivial
instead of requiring a re-read of 30 GB of parquet.

---

## Category C — Image-annotated per tile (annotation ≈ trees; boundary effects only)

Trees were drawn directly on the tiles; there is no untiled master to dedup against. One
annotation ≈ one tree, with only boundary fragmentation to worry about. Box sources are
flagged where tile overlap is unknown.

| Source | Geom | Annotations | GT trees ≈ | Verify? |
|---|---|---:|---:|---|
| SelvaBox / Baudchon et al. 2025 | Boxes | 431,910 | **~83,000** | **RESOLVED — belongs in Category A/B, not C.** Tiles overlap 50 % (train/valid) and 75 % (test), so each crown recurs ~5×. See "SelvaBox resolved" below. |
| OAM-TCD / Veitch-Michaelis et al. 2024 | Boxes | 266,635 | ≈ 266,635 | Partial labels; per-tile. Minor boundary effect. |
| Troles et al. 2024 | Polygons | 92,445 | ≈ 92,445 | Per-tile (coco2048). >50 % rule applies. |
| Radogoshi et al. 2021 | Boxes | 101,837 | ≈ 101,837 | Table says 115,683 (stale). |
| Puliti & Astrup 2025 | Boxes | 48,188 | ≈ 48,188 | Per-tile. |
| WRI | Boxes | 30,971 | ≈ 30,971 | Not in manuscript table? verify. |
| Guangzhou / Sun et al. 2022 | Boxes | 31,228 | ≈ 31,228 | Table has blank count. |
| JustDigIt | Polygons | 27,238 | ≈ 27,238 | Per-tile. |
| Quebec / Lefebvre et al. 2024 | Polygons | 21,626 | **needs check** | 1,322 tiles, no crown ID (only species `taxon_id`). Plantation rows in adjacent tiles may repeat; **flag**. Table says 21,546. |
| Schutte et al. 2025 | Polygons | 18,131 | ≈ 18,131 | Per-tile. |
| Kaggle Palm | Boxes | 18,121 | ≈ 18,121 | Per-tile, patch_overlap=0. |
| Velasquez-Camacho et al. 2023 | Boxes | 14,772 | ≈ 14,772 | Urban. |
| Harz / Lucas et al. 2024 | Polygons | 10,570 | ≈ 10,570 | Per-tile. |
| Zenodo_19695972 / Khan et al. 2026 | Polygons | 8,439 | ≈ 8,439 | Per-tile. |
| Firoze et al. 2025 | Polygons | 5,967 | ≈ 5,967 | Fully annotated crops. |
| Paracou / Ball et al. | Polygons | 5,002 | ≈ 5,002 | Partial. |
| SelvaMask | Polygons | 70,417 | ≈ 70,417 | **Yes** — SelvaBox-derived masks; check overlap with SelvaBox boxes (same trees, two geometries) so trees aren't counted in both the box and polygon totals. |
| TreeCountSegHeight / Li et al. 2023 | Polygons | 4,680 | ≈ 4,680 | Per-tile. |
| DetectTree2 | Polygons | 4,401 | ≈ 4,401 | Per-tile. |
| Takeshige et al. 2025 | Polygons | 4,143 | ≈ 4,143 | Field-verified; per-tile. |
| Ryoungseob / Kwon et al. 2023 | Boxes | 3,827 | ≈ 3,827 | Urban. |
| Zenodo_15155081 / Dumortier 2025 | Boxes | 3,408 | ≈ 3,408 | |
| individual_urban_tree / Zamboni et al. 2021 | Boxes | 3,382 | ≈ 3,382 | Urban. |
| Jansen et al. 2022 | Polygons | 2,779 | ≈ 2,779 | |
| UrbanLondon / Zuniga-Gonzalez 2023 | Polygons | 2,531 | ≈ 2,531 | |
| Kattenborn et al. 2022 | Polygons | 682 | ≈ 682 | Partial. |
| Araujo et al. 2020 | Polygons | 520 | ≈ 520 | |
| OliveTrees / (Spain) | Polygons | 316 | ≈ 316 | |
| Alejandro / Miranda et al. 2021 | Polygons | 794 | ≈ 794 | |
| Wagner et al. 2021 | Polygons | 151 | ≈ 151 | |
| Hickman et al. 2021 | Polygons | 1,264 | ≈ 1,264 | Partial. |
| ReForestTree / Reiersen et al. 2022 | Boxes | 5,890 | ≈ 4,663? | Table says 4,663 — reconcile. |

---

## Category D — Point/box sources that need per-source verification. MEDIUM

| Source | Geom | Annotations | Concern |
|---|---|---:|---|
| **NEON University_of_Florida** (Weecology) | Boxes | 93,126 | 2,151 tiles; `crown_id`/`individual` mostly NaN (only 1,164 individuals labeled). Cannot dedup by ID. Has `box_utm_*` geo columns — **dedup by UTM box across tiles is possible and should be done**. Likely tiling inflation. |
| **AutoArborist / Beery et al. 2022** | Points | 452,366 | Street-inventory points loosely aligned to imagery (train-only, excluded from eval). Table says 452,366. Whether a stem appears in multiple city tiles is unverified. |
| **Ventura et al. 2022** | Points | 96,547 | Urban, per-tile; verify no tile overlap. |
| **Amirkolaee et al. 2023 (UrbanLondon points/TreeFormer)** | Points | 95,067 | Per-tile density points. |
| **Yosemite / Chen & Shang 2022** | Points | 98,949 | Per-tile. |
| **NEON_points** | Points | 51,106 | Table lumps NEON as 99,759 "Points/Boxes"; disaggregate. |
| **Kaggle_LiDAR_RGB / Dubrovin et al. 2024** | Points | 6,938 | |
| **OSBS megaplot / (NEON)** | Points | 6,241 | Field megaplot. |

---

## Category E — Unsupervised / weakly-labeled (EXCLUDE from ground-truth totals)

The loader excludes these by default (`exclude_sources=['*unsupervised*']`). They should
be reported separately and **not** summed into "trees on the ground," because the labels are
model-generated, not verified, and are heavily multi-scan.

| Source | Geom | Annotations | Note |
|---|---|---:|---|
| Weinstein et al. 2018 unsupervised (neon_unsupervised) | Boxes | 7,138,782 | LiDAR-generated over all NEON tiles; the 5,908,313 in Table 1 is a different/older cut — reconcile. Massive multi-tile. |
| OFO 2026 unsupervised (boxes) | Boxes | 110,696 | CHM crown delineation; `treeID`. |
| OFO 2026 unsupervised (points) | Points | 112,719 | `treeID` = 112,719 crowns; multi-mission. |
| Feng et al. 2025 / SPREAD unsupervised | Polygons | 388,725 | Simulation; not real trees at all (Table 1 lists 349,853). |

---

## Concrete recommendations

1. **Add a "ground-truth trees" column** to Table 1 distinct from "annotations," and add a
   footnote defining it (unique physical trees, deduplicated across scans and tiles).
2. **Fix now with code (no human needed):**
   - Cloutier → report one acquisition per zone (~22,860).
   - NEON MultiTemporal → `individual` (784).
   - HARV/Johnson → `crownID` (1,333).
   - BohlmanBCI → `UNIQUE` (1,815).
   - NEON University_of_Florida → dedup by `box_utm_*`.
3. **Flag for human verification before publishing a number:**
   - **OFO field** (mission-level duplication; needs field-inventory dedup).
   - **BCI 2020 vs 2022** (same plot two campaigns — geometric stem matching).
   - **SelvaBox tile overlap** and **SelvaMask ↔ SelvaBox** double representation.
   - **Quebec/Lefebvre** plantation tiles (no crown ID).
   - **AutoArborist** cross-tile stem repeats.
4. **Refresh all stale counts** — several Table 1 numbers no longer match the CSVs.
5. **Report unsupervised sources on a separate line**; never fold Feng/SPREAD (simulation)
   or the 7 M LiDAR boxes into a "trees on the ground" total.

*Generated as an exploratory first pass. Every "≈" number in Categories C/D is the raw
annotation count taken at face value and still needs the boundary/overlap check noted.*
