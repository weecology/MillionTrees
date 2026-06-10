# TreePoints training (TreeFormer)

Fine-tunes the native [DeepForest TreeFormer](https://github.com/jveitchmichaelis/DeepForest/tree/treeformer-training)
point model (PvT backbone + density / OT losses) on the MillionTrees train split.

Install the TreeFormer dependency group (DeepForest `treeformer-training` branch until it
merges to weecology main):

```bash
uv sync --group treeformer
```

Train on the **train** split and evaluate on **test** (`random` or `zeroshot`; zeroshot
holds out entire source datasets from train but still fine-tunes on the rest):

```bash
uv run --group treeformer python training/points/train.py \
  --split-scheme zeroshot \
  --root-dir "$MT_ROOT"
```

Evaluate a saved Lightning checkpoint:

```bash
uv run --group treeformer python training/points/eval.py \
  --checkpoint training/points/outputs/zeroshot/checkpoints/treeformer-epoch=19-val_loss=0.1234.ckpt \
  --split-scheme zeroshot
```

The pretrained (no fine-tuning) TreeFormer baseline lives in
`existing_models/treeformer/eval_points.py`. SLURM training: `training/slurm/train_points.sbatch`.

When point training lands on `weecology/DeepForest` main, drop the git dependency group and
use the released `deepforest` from PyPI.
