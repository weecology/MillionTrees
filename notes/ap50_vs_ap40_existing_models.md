# AP50 vs AP40 (pretrained models, test split)

> **Outcome: AP40 won.** As of 2026-08-04 the datasets score AP at IoU 0.4 only — every table in
> this repo reports AP40 and AP50 is no longer computed. This page is kept as the measurement
> that motivated that switch; its numbers are the v0.21 `outputs_ap_iou/` runs, not the current
> v0.22 leaderboard.

AP50 matches predictions to ground truth at IoU 0.5; AP40 uses IoU 0.4, the same threshold the recall / mask-aware precision behind F1 use. Both are computed on identical predictions in a single eval pass, at each model's leaderboard score threshold. `AP40-AP50` isolates how much of the F1-vs-AP gap is the IoU criterion alone.

## Split: within-distribution

| Task | Model | Recall | Precision | F1 | AP50 | AP40 | AP40-AP50 | F1-AP40 |
|---|---|---|---|---|---|---|---|---|
| TreeBoxes | CanopyRS-DINO-SwinL | 0.635 | 0.731 | 0.680 | 0.411 | 0.462 | +0.051 | +0.218 |
| TreeBoxes | DeepForest-pretrained | 0.359 | 0.588 | 0.446 | 0.143 | 0.207 | +0.064 | +0.239 |
| TreeBoxes | SAM3 | 0.562 | 0.432 | 0.488 | 0.309 | 0.357 | +0.048 | +0.131 |
| TreePolygons | CanopyRS-DINO-SAM3-SelvaMask | 0.762 | 0.874 | 0.814 | 0.313 | 0.375 | +0.062 | +0.439 |
| TreePolygons | Detectree2 | 0.530 | 0.604 | 0.565 | 0.186 | 0.223 | +0.037 | +0.342 |
| TreePolygons | SAM3 | 0.576 | 0.621 | 0.598 | 0.249 | 0.290 | +0.041 | +0.308 |

## Split: out-of-distribution

| Task | Model | Recall | Precision | F1 | AP50 | AP40 | AP40-AP50 | F1-AP40 |
|---|---|---|---|---|---|---|---|---|
| TreeBoxes | CanopyRS-DINO-SwinL | 0.799 | 0.865 | 0.831 | 0.604 | 0.662 | +0.058 | +0.169 |
| TreeBoxes | DeepForest-pretrained | 0.465 | 0.781 | 0.583 | 0.200 | 0.305 | +0.105 | +0.278 |
| TreeBoxes | SAM3 | 0.725 | 0.581 | 0.645 | 0.434 | 0.503 | +0.069 | +0.142 |
| TreePolygons | CanopyRS-DINO-SAM3-SelvaMask | 0.819 | 0.861 | 0.839 | 0.364 | 0.417 | +0.053 | +0.422 |
| TreePolygons | Detectree2 | 0.504 | 0.633 | 0.561 | 0.211 | 0.253 | +0.042 | +0.308 |
| TreePolygons | SAM3 | 0.465 | 0.668 | 0.548 | 0.208 | 0.249 | +0.041 | +0.299 |

Result files:

- `existing_models/canopyrs/outputs_ap_iou/out-of-distribution/results_boxes_out-of-distribution.txt`
- `existing_models/canopyrs/outputs_ap_iou/out-of-distribution/results_polygons_out-of-distribution.txt`
- `existing_models/canopyrs/outputs_ap_iou/within-distribution/results_boxes_within-distribution.txt`
- `existing_models/canopyrs/outputs_ap_iou/within-distribution/results_polygons_within-distribution.txt`
- `existing_models/deepforest/outputs_ap_iou/out-of-distribution/results_boxes_out-of-distribution.txt`
- `existing_models/deepforest/outputs_ap_iou/within-distribution/results_boxes_within-distribution.txt`
- `existing_models/detectree2/outputs_ap_iou/out-of-distribution/results_polygons_out-of-distribution.txt`
- `existing_models/detectree2/outputs_ap_iou/within-distribution/results_polygons_within-distribution.txt`
- `existing_models/sam3/outputs_ap_iou/out-of-distribution/results_boxes_out-of-distribution.txt`
- `existing_models/sam3/outputs_ap_iou/out-of-distribution/results_polygons_out-of-distribution.txt`
- `existing_models/sam3/outputs_ap_iou/within-distribution/results_boxes_within-distribution.txt`
- `existing_models/sam3/outputs_ap_iou/within-distribution/results_polygons_within-distribution.txt`

