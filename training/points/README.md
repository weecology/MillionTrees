# TreePoints training

## TreeFormer (recommended)

Native point detection via [DeepForest TreeFormer](https://github.com/jveitchmichaelis/DeepForest/tree/treeformer-training)
(PvT backbone + density / OT losses). Install the fork:

```bash
uv sync --extra treeformer
```

Fine-tune on MillionTrees (random or zero-shot split):

```bash
uv run --extra treeformer python training/points/train_treeformer.py \
  --split-scheme random \
  --root-dir "$MT_ROOT"
```

Zero-shot benchmark with the released checkpoint (`weecology/deepforest-tree-point`):

```bash
uv run --extra treeformer python docs/examples/baseline_treeformer_points.py \
  --split-scheme random
```

SLURM: `training/slurm/train_treeformer.sbatch` (array over `random` and `zeroshot`).

Evaluate a saved Lightning checkpoint:

```bash
uv run --extra treeformer python training/points/eval_treeformer_checkpoint.py \
  --checkpoint training/points/outputs_treeformer/random/checkpoints/treeformer-epoch=19-val_loss=0.1234.ckpt \
  --split-scheme random
```

When point training lands on `weecology/DeepForest` main, drop the git extra and use PyPI `deepforest` only.

## Legacy pseudo-box training

`train_points.py` fine-tunes RetinaNet on small boxes around each point and converts
predicted box centroids to points at eval. Prefer `train_treeformer.py`; this path
will be removed once TreeFormer leaderboard numbers replace the legacy DeepForest rows.
