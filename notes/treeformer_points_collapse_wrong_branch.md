# TreeFormer point 896 collapse — wrong DeepForest branch (not image size)

**Date:** 2026-07-01
**Symptom:** Overnight point retrain at `image_size=896` (job `36103736`, both splits)
collapsed to `keypoint_acc = 0.000`, `slope = 0.000` on every source — the model
predicts ~zero trees everywhere (counting_mae ~92–106). No crash; it ran all 20 epochs.

## Root cause: shared `.venv` was on the polygon branch, not treeformer-training

The point/polygon workflows share one DeepForest install. `[tool.uv.sources]` in
`pyproject.toml` selects which fork lands in `.venv`, and it had been flipped to the
**polygon Mask R-CNN branch** for the Table 6 work:

```
deepforest = { git = "https://github.com/bw4sz/DeepForest.git",
               branch = "cursor/polygon-maskrcnn-jv-with-safecrop" }   # commit 37bf705
```

That branch also ships a `treeformer.py`, so `training/points/train.py` happily
loaded a `TreeFormerModel` and trained — but its point-training path is broken on
this branch, hence the collapse. Crucially, `srun uv run --group treeformer` does
**not** save you: `[tool.uv.sources]` overrides the `deepforest` source for *every*
group, so `--group treeformer` resolves to the polygon branch anyway.

The last **good** point run (`35991396_1`, 2026-06-30) scored `keypoint_acc = 0.542`
avg / `0.823` max, `slope 0.207` — it ran while the pin was on
`jveitchmichaelis/DeepForest@treeformer-training` (commit `aaad58a`). So **image size
was a red herring; the DeepForest branch is the variable that changed.**

## Fix: a frozen, isolated point venv the polygon workflow can't touch

Built once from the treeformer-training branch, at a separate path, and called
directly (never `uv run`):

```bash
# temporarily flip [tool.uv.sources] deepforest -> jveitchmichaelis@treeformer-training
UV_PROJECT_ENVIRONMENT=.venv-treeformer \
  uv sync --group treeformer --no-group polygon --no-group dev
# then revert the pin (shared .venv stays on the polygon branch)
```

Result: `.venv-treeformer` with `deepforest` @ `treeformer-training` (`aaad58a`),
`torch 2.11.0+cu128`. The point sbatch
`training/slurm/train_points_896_isolated.sbatch` runs `"$VENV/bin/python"` directly
and hard-fails if `direct_url.json` isn't the treeformer-training branch. Because it
never `uv sync`s, a future polygon flip of the shared `.venv` can't clobber it.

## Re-run

`train_points_896_isolated.sbatch` → jobs `36162735_{0,1}` (within / OOD), 896,
pretrained, lr 2e-4, 20 ep. Comet `points-<split>-pretrained-lr2e-4-896-isoenv`.
Outputs `training/points/outputs/<split>_896_isolated/`. If `keypoint_acc` returns
to ~0.5x, 896 is fine and the branch was the sole bug.
