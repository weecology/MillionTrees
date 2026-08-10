# Upstream (pre-tiling) tree counts vs packaged annotation counts

**Method.** For every source in `data_prep/annotation_csvs.cfg`, locate the *original
untiled* vector file (shapefile / GeoPackage / GeoJSON) that the `data_prep` script reads
**before** tiling, and count its features with `pyogrio.read_info`. That feature count is the
number of delineated trees on the ground. Compare it to the packaged v0.20 row count.

Two distinct inflation mechanisms show up:
- **Tiling**: one crown clipped across tile boundaries becomes several rows.
- **Multi-temporal**: the same trees re-annotated on every acquisition/resurvey.

Counts below are `pyogrio` feature counts of the actual files on `/orange`, except Troles
(publisher-reported; we only downloaded the pre-tiled COCO bundle).

---

## A. Multi-temporal — same trees re-annotated per scan/year (VERIFIED)

| Source | Packaged v0.20 | Upstream unique | Evidence |
|---|---:|---:|---|
| **Cloutier et al. 2023** | 159,532 | **22,933** | `Z1/Z2/Z3_polygons.gpkg` are **byte-identical in count across all 7 flight dates** (Z1=10,699, Z2=7,589, Z3=4,645 on every date). One crown map, 7 acquisitions. |
| **Justdiggit 2023** | 27,234 | **18,243** | Per-site/per-year shapefiles: bumila {2018:3589, 2019:3277}, chaludewa {2018:1775, 2019:1786, 2020:2847}, etc. Sum-of-years = 27,238 (= our count exactly). Unique = max per site. |
| **Lucas et al. 2024 (Harz)** | 10,570 | **7,376** | Same plots resurveyed: `test_plot1` = {2009:121, 2016:122, 2022:113}. Unique = max per plot. |
| **OSBS megaplot** | 6,128 | **4,891** | `trees.shp` is *identical* (7,688) in 2014/2016/2017/2018/2019/2021/2023/2025. Use `VisibleTrees_2025` = 4,891. |
| **Vasquez et al. 2023 (BCI)** | 5,160 (2 rows) | **~2,454** | Two crown maps of the same 50 ha ForestGEO plot: 2020=2,454, 2022=2,283. |
| **NEON MultiTemporal** | 2,590 (×3 geoms) | **783** | Per-plot per-year shapefiles (`HARV_002_2018/2019/2022_polygons.shp`); `individual` = 783. |

## B. Tiling inflation — upstream file exists and is smaller (VERIFIED)

| Source | Packaged v0.20 | Upstream unique | Ratio | Upstream file |
|---|---:|---:|---:|---|
| **Troles et al. 2024** | 92,267 | **27,160** | 3.40× | publisher-reported (105 ha, 27,160 crowns); 1024px export = 155,399 vs 2048px = 92,456 |
| **Dubrovin et al. 2024** | 6,925 | **3,602** | 1.92× | `field_survey.geojson` |
| **Frey et al. 2026 (EcoSense TLS)** | 2,813 (×3 geoms) | **1,601** | 1.76× | `crown_polygons_mamba2.gpkg` |
| **Hickman 2021** | 1,264 | **954** | 1.32× | `annotations.shp` (the `*_preds_*.shp` are model output, not GT) |
| **Ball et al. 2023** | 9,388 | **8,546** | 1.10× | paracou `251125_crowns_zenodo2016.gpkg`=4,599 + DetectTree2 Danum 833 / SepilokE 849 / SepilokW 2,265 |
| **Jansen et al. 2023** | 2,777 | **2,547** | 1.09× | 7 plot shapefiles |
| **Reiersen et al. 2022** | 4,899 | **4,699** | 1.04× | `Final_Trees.shp` |

## C. Upstream matches packaged (no inflation — good)

| Source | Packaged | Upstream | Note |
|---|---:|---:|---|
| Wagner et al. 2021 | 151 | **151** | 30 plot gpkgs × ~5 crowns. Exact match. |
| Zuniga-Gonzalez 2023 (UrbanLondon) | 2,482 | 2,516 | train 1,477 + test 1,039 |
| Araujo et al. 2020 | 520 | 535 | `crown_delineation_shapefile.shp` |
| Alejandro/Miranda 2021 | 794 | ~794 | per-plot shp (careful: `*_point_clean.shp` duplicate the polygons) |
| Šrollerů 2025 (Zenodo_15591546) | 32 | 32 | tiny |

## D. No upstream vector file on disk — needs publisher counts. **HIGHEST PRIORITY**

These have no untiled original locally (we downloaded pre-tiled bundles), and together they
dominate the total. Troles proved a pre-tiled source can be inflated 3.4×.

| Source | Packaged v0.20 | Why it matters |
|---|---:|---|
| **SelvaBox** | 431,349 | Largest single source in the benchmark. Pre-tiled by CanopyRS. |
| **OAM-TCD** | 266,623 | Pre-tiled COCO, same format as Troles. |
| **Radogoshi et al. 2021** | 107,173 | |
| **SelvaMask** | 70,273 | |
| **Beery et al. 2022 (AutoArborist)** | 451,007 | Excluded from headline by decision. |
| **Chen & Shang 2022 (Yosemite)** | 98,949 | |
| **Ventura et al. 2022** | 96,509 | |
| **Amirkolaee et al. 2023** | 65,488 | |
| **NEON_points** | 50,562 | |
| **Weecology_University_Florida** | 44,654 | Has `box_utm_*` → dedup possible without publisher. |
| Sun et al. 2022 | 31,225 | |
| WRI | 29,326 | |
| Lefebvre et al. 2024 | 21,546 | |
| Schütte et al. 2025 | 18,124 | |
| Velasquez-Camacho 2023 | 14,771 | |
| Khan et al. 2026 | 8,439 | |
| OFO field 2025 | 136,254 | Multi-mission duplication (86% of rows) — needs field-inventory dedup. |

---

## Revised totals (v0.20, supervised, AutoArborist excluded, cross-geometry deduped)

| Geometry | Packaged annotations | Ground-truth trees |
|---|---:|---:|
| TreeBoxes | 945,022 | ~931,000 |
| TreePoints | 916,030 (465,023 excl. AutoArborist) | ~355,000 |
| TreePolygons | 453,842 | ~229,000 |
| **Unique physical trees** | | **~1,515,000** |

Cross-geometry: Allen et al. 2025 (1,609) and NEON MultiTemporal (783) are counted **once**
(in boxes); Frey/EcoSense = 1,601 once. SelvaMask and SelvaBox were verified to share **zero
orthomosaics**, so both are counted.

## Biggest remaining risk to the total
SelvaBox (431k) + OAM-TCD (267k) = **~698k of the ~931k box total**, both pre-tiled with no
local original. If either behaves like Troles (3.4×), the headline drops materially. Getting
their publisher-reported crown counts is the single highest-value remaining check.
