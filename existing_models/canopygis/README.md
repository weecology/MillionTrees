# CanopyGIS v1.8.1 baseline

Pure zero-shot planar tree-crown polygon delineation baseline from
[CanopyGIS v1.8.1](https://github.com/napattan/CanopyGIS/tree/v1.8.1),
evaluated on MillionTrees `TreePolygons`.

| Script | Task | Weights |
|---|---|---|
| `eval_polygons.py` | TreePolygons | [`facebook/sam3`](https://huggingface.co/facebook/sam3) (pure zero-shot, `0` forestry fine-tuning) |

## Pipeline Overview

CanopyGIS v1.8.1 is a pure zero-shot planar pipeline requiring no forestry
fine-tuning, no auxiliary LiDAR/DSM layers, and no external GSD metadata
(`gsd_m=None` throughout evaluation):

1. Zero-shot `facebook/sam3` canopy mask proposals on RGB imagery.
2. Biophysical shadow-crevice geodesic watershed crown separation.
3. 1D LUT Region Adjacency Graph (RAG) over-segmentation refinement.
4. Symmetric Voronoi non-overlapping planar partition guaranteeing
   `overlap_px == 0` and single-component topology (`Euler == 1`).

Peak GPU memory stays under `< 5 GB` VRAM, and `eval_polygons.py` streams
metrics batch-by-batch via `TreePolygonsStreamingEvalState`.

## Setup

This directory is an isolated `uv` environment mirroring
`existing_models/sam3`:

```bash
cd existing_models/canopygis
uv sync
```

SAM 3 (`facebook/sam3`) is gated on Hugging Face: request access on the model
page and authenticate with `huggingface-cli login` (or export `HF_TOKEN` /
pass `--hf-token`) before running `eval_polygons.py`.

## Run

Out-of-distribution test split (`v0.25`, `4,133` tiles):

```bash
uv run python eval_polygons.py \
  --root-dir "$MT_ROOT" \
  --version 0.25 \
  --split-scheme out-of-distribution \
  --device cuda \
  --score-threshold 0.10 \
  --image-size 1024 \
  --output-dir outputs/out-of-distribution \
  --viz-dir outputs/out-of-distribution/viz_polygons \
  --viz-n-per-source 2 \
  --hf-token "$HF_TOKEN"
```

Within-distribution test split (`v0.25`, `1,623` tiles):

```bash
uv run python eval_polygons.py \
  --root-dir "$MT_ROOT" \
  --version 0.25 \
  --split-scheme within-distribution \
  --device cuda \
  --score-threshold 0.10 \
  --image-size 1024 \
  --output-dir outputs/within-distribution \
  --viz-dir outputs/within-distribution/viz_polygons \
  --viz-n-per-source 2 \
  --hf-token "$HF_TOKEN"
```

Held-out TLS validation split (`v0.25`, `59` complete tiles):

```bash
uv run python eval_polygons.py \
  --root-dir "$MT_ROOT" \
  --version 0.25 \
  --split-scheme out-of-distribution \
  --eval-split validation \
  --complete-tiles-only \
  --device cuda \
  --score-threshold 0.10 \
  --image-size 1024 \
  --output-dir outputs/validation \
  --viz-dir outputs/validation/viz_polygons \
  --save-predictions outputs/validation/preds_polygons.pkl \
  --hf-token "$HF_TOKEN"
```

## Official v0.25 Results

Evaluated on MillionTrees `TreePolygons` **v0.25** (`image_size=1024`,
`resolution_mode=direct`, `gsd_m=None`, `score_threshold=0.10`, `0` forestry
fine-tuning):

| Split (`v0.25`) | Tiles | MaskRecall | MaskAwarePrecision | F1 | MaskAccuracy | AP40 | AP60 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `out-of-distribution` (test) | 4,133 | 0.617 | 0.936 | 0.744 | 0.314 | 0.439 | 0.350 |
| `within-distribution` (test) | 1,623 | 0.459 | 0.882 | 0.604 | 0.208 | 0.243 | 0.156 |
| `validation` (`complete-59`) | 59 | 0.463 | 0.970 | 0.627 | 0.316 | 0.329 | 0.118 |

*(Note: `validation` `AP40 / AP60` above are `eval_polygons.py`'s direct pooled scores; rescoring the saved `.pkl` with `scripts/rescore_validation_ap.py --max-margin 0.025` gives `0.354 / 0.160` 2-source macro.)*

Across all `5,815` tile evaluations (`458,851` predicted crowns), predicted masks
satisfy `overlap_px == 0`, `empty_masks == 0`, `holes == 0`, and `Euler == 1`.

Useful options:

- `--num-workers`: DataLoader worker count (defaults to `0` on Windows `spawn`
  to avoid duplicating polygon metadata in worker RAM, and `2` on POSIX/SLURM).
- `--resolution-mode native`: read raw RGB tiles from disk at native
  resolution and resize predicted masks to `--image-size` (default: `direct`).
- `--no-remap-ws-scores`: disable watershed score remapping.
- `--mini --max-batches 2`: quick smoke test on the mini release.
- `--sweep`: run a threshold sweep (`--score-threshold 0.01`) writing
  `<output-dir>/threshold_sweep.csv`.

Test runs write `outputs/<split>/results_polygons_<split>.{txt,json}` for
`scripts/make_benchmark_table.py`; non-test splits write
`results_polygons_<eval_split>.{txt,json}` (e.g.
`results_polygons_validation.txt`).
