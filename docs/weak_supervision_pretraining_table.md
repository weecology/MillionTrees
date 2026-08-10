# Weak-supervision box pretraining: polygon Mask R-CNN results

Evaluated on TreePolygons v0.13 (1,473 test images, 16 sources).

## Random split

| Init | Epoch | Accuracy | Recall | mAP | Maskaware precision |
|---|---:|---:|---:|---:|---:|
| COCO baseline | 39 (full) | 0.224 | 0.379 | 0.040 | 0.763 |
| Box-pretrained (TreeBoxes + unsupervised) | 4 (early) | **0.373** | **0.684** | **0.071** | **0.844** |
| **Δ (box − coco)** | | **+0.149** | **+0.305** | **+0.031** | **+0.081** |

**Note:** Box-pretrained checkpoint is from epoch 4 of the v0.13 run (training cancelled to refit data loaders; full 40-epoch run is in progress). The COCO baseline is a completed 40-epoch run on the same v0.13 data. Even at epoch 4, box-pretrained backbone produces a large gain across all metrics.

## Zeroshot split

Pending — eval job `32651194_1` and full training runs in progress.

---

*Last updated 2026-05-18. Source data:*
- *Box-pretrained random:* `training/weak_supervision/outputs/eval_current/polygon_box_pretrained/random/results_random.txt` (checkpoint `polygons-epoch=04-val_loss=-7045186.0000.ckpt`)
- *COCO random:* `training/weak_supervision/outputs/polygon_coco/random/results_random.txt` (checkpoint `polygons-epoch=39-val_loss=-465924.2500.ckpt`)
