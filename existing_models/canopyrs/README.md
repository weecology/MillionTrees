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

This is a uv env, like `existing_models/detectree2`. The only pieces uv/pip can't
resolve are CanopyRS itself plus detrex (DINO's `ms_deform_attn` CUDA op) and
Detectron2, which are compiled from source against the installed torch — so those,
along with the torch/CUDA build pins, are kept out of `pyproject.toml` and recorded
in [`detrex.toml`](detrex.toml). The scripted build is [`build_env.sh`](build_env.sh):

```bash
bash existing_models/canopyrs/build_env.sh
```

Or the equivalent steps by hand (see `detrex.toml` for the exact pins):

```bash
git clone https://github.com/hugobaudchon/CanopyRS.git
git -C CanopyRS submodule update --init --recursive   # detrex (+ its Detectron2 fork)

cd existing_models/canopyrs
uv venv --python 3.10 && source .venv/bin/activate
module load cuda/12.8.1 gdal/3.7.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 FORCE_CUDA=1   # L4/sm_89; FORCE_CUDA: login node has no GPU

uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install "GDAL==$(gdal-config --version)"
uv pip install --no-build-isolation -e ../../../CanopyRS/detrex/detectron2 -e ../../../CanopyRS/detrex
uv pip install -e ../../../CanopyRS                       # the canopyrs package
uv pip install -e .                                       # milliontrees + SAM 3 transformers
```

SAM 3 (`facebook/sam3`) is gated: request access on the model page and
`huggingface-cli login` (or set `HF_TOKEN` / pass `--hf-token`) before running
`eval_polygons.py`.

## Run

```bash
python eval_boxes.py    --split-scheme within-distribution   --device cuda --output-dir outputs/within-distribution
python eval_polygons.py --split-scheme within-distribution   --device cuda --output-dir outputs/within-distribution --hf-token "$HF_TOKEN"
```

Add `--viz-dir <dir>` for per-source prediction overlays, `--mini` for a smoke test, and
`--split-scheme out-of-distribution` for the held-out-source task. Results are written to
`outputs/<split>/results_<geometry>_<split>.{txt,json}`, which
`scripts/make_benchmark_table.py` reads to regenerate the leaderboard.
