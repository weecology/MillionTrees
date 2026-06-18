# Models and repository structure

Every leaderboard number is produced by a script in one of two top-level folders.
The split is deliberate and maps directly onto the **Fine-tuned** column of the
[leaderboard](leaderboard.md):

| Folder | Purpose | Leaderboard column | Data used |
|---|---|---|---|
| `training/` | Models **trained** on the MillionTrees train split, then evaluated on the test split | Fine-tuned ✓ | train + test |
| `existing_models/` | **Pretrained** released weights evaluated against the MillionTrees test split | Fine-tuned ✗ | test only |

There are no model scripts in `docs/examples/`. If you are looking for a runnable
template, see `existing_models/external_segmentation_adapter.py`.

## `training/` — fine-tuned models (✓)

One folder per geometry, each with the same two entry points:

| Geometry | Model | Train | Evaluate a checkpoint |
|---|---|---|---|
| `training/boxes/` | DeepForest (RetinaNet) | `train.py` | `eval.py` |
| `training/points/` | TreeFormer | `train.py` | `eval.py` |
| `training/polygons/` | Mask R-CNN | `train.py` | `eval.py` |

Common usage (works for `within-distribution` and `out-of-distribution` split schemes):

```bash
uv run python training/boxes/train.py --split-scheme within-distribution --root-dir "$MT_ROOT"
```

The point model needs the TreeFormer extra (DeepForest
[`treeformer-training`](https://github.com/jveitchmichaelis/DeepForest/tree/treeformer-training)
branch until it merges to weecology main):

```bash
uv sync --group treeformer
uv run --group treeformer python training/points/train.py --split-scheme within-distribution
```

Each run writes `training/<geometry>/outputs/<split>/results_<split>.txt` (+ `.json`),
which `scripts/make_benchmark_table.py` reads to regenerate the leaderboard tables.

## `existing_models/` — pretrained baselines (✗)

One folder per model, each containing `eval_<geometry>.py` for the geometries that
model natively predicts. Each model folder has its own `pyproject.toml` so its
dependencies stay isolated from the core package.

| Model | Folder | Geometries |
|---|---|---|
| DeepForest | `existing_models/deepforest/` | boxes |
| TreeFormer | `existing_models/treeformer/` | points |
| SAM3 | `existing_models/sam3/` | boxes, points, polygons |
| CanopyRS (DINO Swin-L / DINO+SAM3) | `existing_models/canopyrs/` | boxes, polygons |

```bash
uv run python existing_models/deepforest/eval_boxes.py --split-scheme out-of-distribution --root-dir "$MT_ROOT"
```

Results are written to `existing_models/<model>/outputs/<split>/results_<geometry>_<split>.txt`.

`existing_models/external_segmentation_adapter.py` is a template showing how to convert
an arbitrary external model's outputs into the MillionTrees evaluation format; copy it as
the starting point for a new `existing_models/<your_model>/` entry.

## Reproducing the leaderboard for a new dataset version

SLURM launchers fan out over geometry × split. To launch everything after packaging a
new dataset version:

```bash
# 1. fine-tuned training jobs + pretrained eval jobs
bash slurm/submit_all.sh

# 2. once all jobs finish, regenerate the tables
uv run python scripts/make_benchmark_table.py --splits within-distribution out-of-distribution
```

`slurm/submit_all.sh` simply calls the two per-folder launchers, which you can also run
independently:

- `training/slurm/submit_all_training.sh` → `train_boxes.sbatch`, `train_points.sbatch`, `train_polygons.sbatch`
- `existing_models/slurm/submit_all_eval.sh` → `eval_deepforest.sbatch`, `eval_treeformer.sbatch`, `eval_sam3.sbatch`, `eval_canopyrs.sbatch`

For a dependency-chained run that automatically rebuilds the table once every job
finishes, use `slurm/run_benchmark.sbatch` instead.

## Leaderboard panel figures (fine-tuned)

The images embedded in [leaderboard.md](leaderboard.md) (`leaderboard_predictions_*.png`)
are **not** produced by `submit_all.sh`. They are regenerated from **fine-tuned checkpoints**
after training completes:

| Geometry | Model | Checkpoint path |
|---|---|---|
| TreePoints | TreeFormer | `training/points/outputs/<split>/checkpoints/` |
| TreeBoxes | DeepForest | `training/boxes/outputs/<split>/checkpoints/` |
| TreePolygons | Mask R-CNN | `training/polygons/outputs/<split>/checkpoints/` |

Each figure has two rows (**within-distribution**, **out-of-distribution** fine-tuning tasks) and two columns
(ground truth vs fine-tuned prediction on the same test image).

```bash
uv run --group treeformer python scripts/create_finetuned_visualizations.py \
  --root-dir "$MT_ROOT" \
  --output-dir docs \
  --panel-dir docs/figures/finetuned_panels
```

Outputs:

- `docs/leaderboard_predictions_{points,boxes,polygons}.png` and `.svg` (combined panels)
- `docs/figures/finetuned_panels/<geometry>_<split>_{ground_truth,finetuned}.svg` (one file per panel for manuscript layout)

On SLURM: `sbatch slurm/visualize_finetuned.sbatch` (included as a dependent step in
`run_benchmark.sbatch` after the three training array jobs).
