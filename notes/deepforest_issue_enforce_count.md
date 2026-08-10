# `enforce_count=True` makes the TreeFormer count loss identically zero

## Summary

With the default `enforce_count=True`, `_normalize_density` rescales the density map during
training so its sum **exactly equals the ground-truth count**. The `count` loss then compares a
value to itself, so it is identically `0.0` with a zero gradient — the count head is never
trained, at any learning rate, for any number of epochs.

Because `enforce_count` falls back to the (consequently untrained) CLS head at inference, the
predicted density is scaled by a garbage count at test time.

## Repro

```python
import torch
from deepforest.models.treeformer import TreeFormerModel

m = TreeFormerModel()          # defaults: enforce_count=True
m.train()

B, H, W = 2, 32, 32
points = [torch.rand(n, 2) * 128 for n in (37, 91)]
gt_counts = torch.tensor([float(len(p)) for p in points])
score_map = (torch.rand(B, 1, H, W) * 1000).requires_grad_(True)   # deliberately mis-scaled
cls_count = torch.rand(B) * 5

density_map, normed = m._normalize_density(score_map, cls_count, gt_count=gt_counts)
out = m.compute_loss(
    density_maps=[density_map] * 3,
    normed_density=normed,
    cls_outputs=[cls_count.view(B, 1)] * 3,
    targets=[{"points": p} for p in points],
    image_shapes=[(128, 128)] * B,
)

print("density_map sums:", density_map.view(B, -1).sum(1).tolist())
print("count_loss      :", float(out["count_loss"]))
print("count_mae       :", float(out["count_mae"]))
print("grad            :", float(torch.autograd.grad(
    out["count_loss"], score_map, allow_unused=True)[0].abs().sum()))
```

**Output:**

```
density_map sums: [37.0, 91.0]      <-- forced to equal GT, from raw sums of ~5e5
count_loss      : 0.0
count_mae       : 0.0
grad            : 0.0
```

## Cause

`models/treeformer.py`:

```python
if self.enforce_count:
    if self.training and gt_count is not None:
        count = gt_count.view(B, 1, 1, 1).clamp(min=1e-4)     # train: GT count
    else:
        count = cls_count.view(B, 1, 1, 1).abs().clamp(min=1e-4)  # inference: CLS head
    return normed * count, normed
```

`normed` sums to 1, so `density_map.sum() == gt_count` exactly, and

```python
count_loss = cls_l1(torch.log1p(pred_sum), torch.log1p(point_counts))   # |x - x| = 0
```

`enforce_count: bool = True` is the default in both `models/treeformer.py` and `conf/schema.py`.
`conf/point.yaml` does not override it, so anyone fine-tuning from that config hits this;
`conf/point_pretrain.yaml` does set `enforce_count: false`, which is why the released
pretrained weights are unaffected.

## Impact observed downstream

Fine-tuning the released checkpoint on ~1M points, 20 epochs, `val_loss` falling monotonically
every epoch (0.4010 → 0.3620) — but `val_loss` is *exactly* `val_density_l1_loss` at every
epoch, and counting calibration degrades relative to the pretrained starting point:

| | pretrained | fine-tuned |
|---|---|---|
| macro counting nMAE | 0.489 | 0.901 |
| counting slope | 0.158 | −0.922 |
| held-out MAE (trees/image) | 46.9 | 302.2 |

Localization improves (recall 0.76 → 0.85) while precision collapses (0.74 → 0.54), i.e. the
model learns *where* trees are but never *how many*, and over-predicts ~5×.

## Suggested fix

Skip or guard the `count` term when `enforce_count` is on, since it cannot contribute a
gradient — or don't use the GT count for the training-path rescale. At minimum, warn when
`enforce_count=True` and `"count" in losses`, as that combination silently trains nothing.
