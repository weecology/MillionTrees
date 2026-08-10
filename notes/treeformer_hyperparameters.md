# TreeFormer point-detection hyperparameters

How TreeFormer turns an image into points, and which knob controls each stage.
Code lives in the `treeformer-training` DeepForest branch:
`deepforest/models/treeformer.py` (model) and `deepforest/utilities.py::density_to_points`
(peak extraction). The MillionTrees eval metric adds two more knobs on top.

## The pipeline (image → points)

```
image (B,3,H,W)
  └─ PvT-V2 backbone ──> features at stride 4   (downsample_ratio = 4)
       └─ Regression head ──> raw density map  out_L[0]  (B,1,H/4,W/4)
            └─ _normalize_density(., cls_count) ──> density map scaled to predicted count
                 └─ density_to_points(score_thresh, score_integration_radius)
                      ├─ normalize map to [0,1] per image
                      ├─ peak_local_max(min_distance=radius, threshold_rel=score_thresh)
                      └─ scores = normalized density at each peak
                           └─ postprocess_density: scale peak (x,y) by 4 back to image px
                                └─ MillionTrees metric: keep scores > eval_score_threshold,
                                   match GT within per-source GSD distance threshold
```

The density map is at **1/4 image resolution**. For a 448 px crop the map is
112×112, so the finest separable structure is ~4 image px. This is a hard
ceiling: two trees whose centers fall in the same 4 px cell can never produce
two peaks, no matter how the downstream knobs are set.

## Inference / post-processing knobs (change predictions without retraining)

| Knob | Where set | Default | Std | Effect | Raise → | Lower → |
|---|---|---|---|---|---|---|
| `score_integration_radius` | `model.model.*` (submodule) | 5 | **2** | `peak_local_max` `min_distance` in **density-map px** (×4 in image px). Minimum spacing between detections. | fewer, more separated points; better counting MAE | more, closer points; r1 over-predicts (duplicates) → counting MAE regresses |
| `score_thresh` | `model.model.*` (submodule) | 0.5 (cfg 0.1) | **0.1** | `peak_local_max` `threshold_rel`: drop peaks below `score_thresh × max`. | drops low/marginal peaks (recall↓, precision↑) | keeps marginal peaks. **Near no-op** below ~0.1: the density map is peaky, few marginal peaks exist there, and the metric's `eval_score_threshold` is the binding filter. |
| `enforce_count` | model config | True | True | Rescales the density map so its sum = the CLS head's predicted count, setting absolute peak heights (hence the scores the metric thresholds). | — | raw unscaled map |

**Gotcha (verified):** `postprocess_density` reads `model.model.score_thresh` /
`model.model.score_integration_radius`, frozen on the submodule at construction.
Setting `model.config.*` after `load_model()` is silently ignored. Always set on
the submodule, and **re-set after `load_from_checkpoint`** (it rebuilds the
submodule from config → reverts to deepforest defaults).

## Eval-metric knobs (change scoring, not predictions)

| Knob | Where | Default | Effect |
|---|---|---|---|
| `eval_score_threshold` | `TreePointsDataset` (=0.4) | 0.4 | **The binding score filter.** Points with score ≤ 0.4 are discarded before matching. TreeFormer scores are normalized density ∈ [0,1], so peaks added by lowering `score_thresh` from 0.1 are detected then thrown away here. Lowering this is the untested *direct* recall lever. |
| per-source distance threshold | `SOURCE_GSD` + `real_world_threshold_m=3.0` | 3 m | Match radius = `3 m / (gsd · native_crop_px)`, then ×448. Ranges ~3–88 px across sources for the same 3 m. Missing GSD → flat `distance_threshold=0.02` fallback. |

## Training knobs (change the learned density map — the deeper lever)

| Knob | Default | Effect on close-together detections |
|---|---|---|
| **`density_sigma`** | 5.0 (ctor); **3.0 in the released `weecology/deepforest-tree-point` checkpoint** — restored from its HF config on load, and inherited by finetuning unless overridden | Gaussian σ (in density-map px) used to build the GT density target. **This is the most likely culprit for merged peaks.** σ=3 on the 1/4-res map ≈ 12 image px: each tree becomes a wide blob, so two trees within ~2σ have overlapping GT blobs and the model is *trained* to emit one merged mode. `min_distance` cannot separate what the map already merged. Lowering σ → sharper, more separable peaks (but noisier counts / harder optimization). Not currently a CLI arg — to test it, set `model.model.density_sigma` before `trainer.fit`. |
| `mae_weight` | 1.0 | Weight on total-count MAE loss (density-sum vs GT count, log space). |
| `ot_weight` | 0.1 | Weight on optimal-transport loss (spatial placement of mass). |
| `density_l1_weight` | 0.01 | Weight on pixelwise L1 vs the Gaussian GT density. Higher → map hews closer to the (σ-blurred) GT shape. |
| `count_cls_weight` | 1.0 | Weight on the CLS count-density head (used for `enforce_count` at inference). |
| `num_of_iter_in_ot` / `sinkhorn_reg` | 100 / 1.0 | Sinkhorn solver iters / regularization for the OT loss. |
| `norm_cood` | False | Normalize coordinates in the OT loss. |
| `backbone` | `pvt_v2_b3` | PvT-V2 variant. Larger → more capacity, same stride-4 ceiling. |
| `lr`, `epochs` | 1e-5 / 20 | Optimization. ~1e-5 for finetuning pretrained, ~2e-4 for random-weights. |

## So why are close-together predictions suppressed?

Three stacked ceilings, in order of how deep they sit:

1. **`density_sigma=3` (train-time blob width; released checkpoint).** Likely the dominant one. The
   model is trained to merge nearby trees into one broad blob. **Deepest lever —
   needs retraining.**
2. **`downsample_ratio=4` (resolution).** 1/4-res map can't separate sub-4 px
   trees. Architectural; not a config knob.
3. **`score_integration_radius` (peak spacing) + `eval_score_threshold=0.4`
   (score cut).** Post-hoc; cheap to sweep but can't recover peaks the map never
   formed. r2 is the current sweet spot; r1 regresses counting MAE.

Sweep order to find the right pieces: confirm (3) is saturated on a fixed
checkpoint (use `visualize_density.py` to see whether close trees even form
separate modes), then attack (1) by retraining with smaller `density_sigma`.
