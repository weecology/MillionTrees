# CanopyRS baselines

Pretrained tropical tree-crown baselines from the SelvaBox / SelvaMask papers
([CanopyRS](https://github.com/hugobaudchon/CanopyRS)), evaluated on the MillionTrees
test split.

| Script | Task | Weights |
|---|---|---|
| `eval_boxes.py` | TreeBoxes | [`CanopyRS/dino-swin-l-384-multi-NQOS`](https://huggingface.co/CanopyRS/dino-swin-l-384-multi-NQOS) (DINO Swin-L, multi-resolution multi-dataset) |
| `eval_polygons.py` | TreePolygons | [`CanopyRS/dino-swin-l-384-multi-NQOS-selvamask-FT`](https://huggingface.co/CanopyRS/dino-swin-l-384-multi-NQOS-selvamask-FT) detector + SelvaMask fine-tuned SAM 3 segmenter |

The detector proposes crown boxes; for polygons SAM 3 turns each box into an instance
mask (the SelvaMask segmentation pipeline). Both run through CanopyRS' own model
wrappers, driven per image over the MillionTrees test loader.

## Setup

CanopyRS depends on detrex + Detectron2, which must be built from source, so this folder
cannot be provisioned with a plain `uv sync`. Install CanopyRS first (its conda recipe is
the supported path), then install MillionTrees into the same environment:

```bash
# 1. Install CanopyRS per its installation guide:
#    https://hugobaudchon.github.io/CanopyRS/getting-started/installation/
git clone https://github.com/hugobaudchon/CanopyRS.git
cd CanopyRS
conda create -n canopyrs -c conda-forge python=3.10 mamba && conda activate canopyrs
mamba install gdal=3.6.2 -c conda-forge
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
git submodule update --init --recursive
pip install -e .
pip install --no-build-isolation -e ./detrex/detectron2 -e ./detrex
# SAM 3 is required for eval_polygons.py and needs transformers>=5.0.0rc1
pip install "transformers>=5.0.0rc1"

# 2. Install MillionTrees into the same env
pip install -e /path/to/MillionTrees
```

SAM 3 (`facebook/sam3`) is gated: request access on the model page and
`huggingface-cli login` (or pass `--hf-token`) before running `eval_polygons.py`.

## Run

```bash
python eval_boxes.py    --split-scheme random   --device cuda --output-dir outputs/random
python eval_polygons.py --split-scheme random   --device cuda --output-dir outputs/random --hf-token "$HF_TOKEN"
```

Add `--viz-dir <dir>` for per-source prediction overlays, `--mini` for a smoke test, and
`--split-scheme zeroshot` for the held-out-source task. Results are written to
`outputs/<split>/results_<geometry>_<split>.{txt,json}`, which
`scripts/make_benchmark_table.py` reads to regenerate the leaderboard.
