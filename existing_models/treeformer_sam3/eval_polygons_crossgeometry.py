"""
TreeFormer -> SAM3 cross-geometry polygon evaluation on MillionTrees.

Pipeline:
  1. Load TreePolygons with crossgeometry split.
  2. For each image, run DeepForest TreeFormer to predict tree centroids.
  3. Pass those centroids as SAM3 point prompts (no text prompt) to generate polygon masks.
  4. Evaluate polygon masks against ground-truth polygon annotations.

Visualization (--viz-dir):
  Purple  = ground-truth polygon masks
  Orange  = SAM3 predicted masks
  Cyan    = TreeFormer point predictions
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from rasterio.features import rasterize as rio_rasterize

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.datasets.polygon_stream_eval import TreePolygonsStreamingEvalState
from milliontrees.common.eval_visualization import (
    _blend_masks,
    _draw_points,
    _resize_eval_image,
    _to_numpy_image,
    _tensor_masks,
    _COLOR_GROUND_TRUTH,
    _COLOR_PREDICTION,
)

_COLOR_TF_POINTS = (0, 200, 255)  # cyan

TREEFORMER_CONFIG = {
    "architecture": "treeformer",
    "model": {
        "name": "weecology/deepforest-tree-point",
        "revision": "main",
    },
    "patch_size": 512,
    "patch_overlap": 0.1,
    "score_thresh": 0.3,
    "point": {
        "backbone": "pvt_v2_b3",
        "score_integration_radius": 5,
        "nms_distance_thresh": 5.0,
        "distance_threshold": 10.0,
    },
}


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def to_numpy_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float [0,1] tensor to (H, W, C) uint8 numpy array."""
    return (image_tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def _empty_pred(height: int, width: int) -> dict:
    return {
        "y": torch.zeros((0, height, width), dtype=torch.bool),
        "labels": torch.zeros((0,), dtype=torch.int64),
        "scores": torch.zeros((0,), dtype=torch.float32),
    }


def polygons_to_prediction(poly_gdf, height: int, width: int) -> dict:
    """Rasterize shapely polygon geometries from a GeoDataFrame into boolean masks."""
    if poly_gdf is None or len(poly_gdf) == 0:
        return _empty_pred(height, width)

    masks: List[np.ndarray] = []
    scores_list: List[float] = []
    for _, row in poly_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        mask = rio_rasterize(
            [(geom.__geo_interface__, 1)],
            out_shape=(height, width),
            fill=0,
            dtype=np.uint8,
        ).astype(bool)
        masks.append(mask)
        score = float(row["score"]) if "score" in poly_gdf.columns and pd.notna(row["score"]) else 1.0
        scores_list.append(score)

    if not masks:
        return _empty_pred(height, width)

    return {
        "y": torch.tensor(np.stack(masks), dtype=torch.bool),
        "labels": torch.zeros(len(masks), dtype=torch.int64),
        "scores": torch.tensor(scores_list, dtype=torch.float32),
    }


def _slug(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(name), flags=re.UNICODE).strip("_")
    return s[:max_len] if len(s) > max_len else s or "source"


def save_viz_with_tf_points(
    dataset,
    y_pred: List[dict],
    y_true: List[dict],
    tf_points: List[torch.Tensor],
    metadata: torch.Tensor,
    out_dir: str,
    n_per_source: int = 4,
    score_threshold: float = 0.5,
    viz_size: int | None = None,
) -> None:
    """Write overlay PNGs: purple=GT masks, orange=SAM3 masks, cyan=TreeFormer points."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not isinstance(metadata, torch.Tensor):
        metadata = torch.as_tensor(metadata)

    eval_size = int(getattr(dataset, "image_size", 224))
    image_size = viz_size if viz_size is not None else eval_size
    ma = metadata if isinstance(metadata, torch.Tensor) else torch.as_tensor(metadata)
    fid_to_ds_idx = {int(ma[i, 0].item()): i for i in range(ma.shape[0])}

    per_source_count: dict[int, int] = {}

    for row in range(len(y_pred)):
        source_id = int(metadata[row, 1].item())
        if per_source_count.get(source_id, 0) >= n_per_source:
            continue

        filename_id = int(metadata[row, 0].item())
        ds_idx = fid_to_ds_idx[filename_id]

        raw = _to_numpy_image(dataset.get_input(ds_idx))
        # Blend masks at eval_size (matches y_pred/y_true resolution)
        rgb = _resize_eval_image(raw, eval_size)
        gt_m = _tensor_masks(y_true[row].get("y"), eval_size)
        pm = _tensor_masks(y_pred[row].get("y"), eval_size)
        scores = y_pred[row].get("scores")
        if scores is not None and pm.shape[0] > 0:
            sc = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores, dtype=torch.float32)
            pm = pm[sc.detach().cpu().float().numpy() > score_threshold]

        arr = _blend_masks(rgb, gt_m, _COLOR_GROUND_TRUTH, alpha=0.35)
        arr = _blend_masks(arr, pm, _COLOR_PREDICTION, alpha=0.35)

        # Upscale to viz_size for the final PNG
        base = Image.fromarray(arr, mode="RGB")
        if image_size != eval_size:
            base = base.resize((image_size, image_size), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(base)

        pts = tf_points[row]
        if pts is not None and len(pts) > 0:
            pts_np = pts.detach().cpu().float().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts)
            # Scale point coords from eval_size to viz output size
            scale = image_size / eval_size
            pts_np = pts_np * scale
            _draw_points(draw, pts_np, _COLOR_TF_POINTS, r=max(4, int(4 * scale)))

        src_name = getattr(dataset, "_source_id_to_code", {}).get(source_id, str(source_id))
        stem = Path(getattr(dataset, "_filename_id_to_code", {}).get(filename_id, f"id_{filename_id}")).stem
        sub = out_path / _slug(src_name)
        sub.mkdir(parents=True, exist_ok=True)
        k = per_source_count.get(source_id, 0)
        base.save(sub / f"{k:03d}_{_slug(stem)}.png")
        per_source_count[source_id] = k + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TreeFormer -> SAM3 cross-geometry polygon eval on MillionTrees."
    )
    parser.add_argument(
        "--root-dir", type=str,
        default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--split-scheme", type=str, default="crossgeometry",
        choices=["crossgeometry"],
        help="Only crossgeometry makes sense for this model.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="SAM2 IoU threshold for filtering predicted masks")
    parser.add_argument("--mask-threshold", type=float, default=0.3,
                        help="SAM2 mask binarization threshold")
    parser.add_argument("--prompt-batch-size", type=int, default=32,
                        help="Number of point/box prompts processed per SAM2 forward pass")
    parser.add_argument("--negative-prompts", action="store_true", default=True,
                        help="Pass all other detections as negative SAM2 prompts per instance (default on)")
    parser.add_argument("--no-negative-prompts", dest="negative_prompts", action="store_false",
                        help="Disable negative point prompts")
    parser.add_argument("--max-point-prompts", type=int, default=None,
                        help="Max negative prompts per instance (nearest-neighbour fallback)")
    parser.add_argument("--score-thresh-tf", type=float, default=0.3,
                        help="TreeFormer peak-score threshold")
    parser.add_argument("--nms-distance-thresh", type=float, default=5.0,
                        help="TreeFormer NMS merge radius (px); lower keeps more nearby detections")
    parser.add_argument("--patch-size", type=int, default=0,
                        help="If >0, run predict_tile on original images at this patch size "
                             "(no overlap) instead of predict_step on resized batch tensors")
    parser.add_argument("--patch-fraction", type=float, default=0.0,
                        help="If >0, run predict_tile with patch_size = orig_dim * patch_fraction "
                             "per image (e.g. 0.5 ≈ 4 patches). Overrides --patch-size.")
    parser.add_argument("--sam2-model", type=str, default="facebook/sam2.1-hiera-small",
                        help="HuggingFace model ID for SAM2")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source overlay PNGs "
                             "(purple=GT, orange=SAM3 masks, cyan=TreeFormer points)")
    parser.add_argument("--viz-n-per-source", type=int, default=4)
    parser.add_argument("--viz-size", type=int, default=512,
                        help="Output PNG resolution (pixels); independent of --image-size used for eval")
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    from deepforest import main as df_main

    config_args = dict(TREEFORMER_CONFIG)
    config_args["score_thresh"] = args.score_thresh_tf
    config_args["point"] = dict(TREEFORMER_CONFIG["point"])
    config_args["point"]["nms_distance_thresh"] = args.nms_distance_thresh
    model = df_main.deepforest(config_args=config_args)
    model.eval()

    device = select_device(args.device)

    dataset = get_dataset(
        "TreePolygons",
        root_dir=args.root_dir,
        download=args.download,
        mini=args.mini,
        split_scheme=args.split_scheme,
        image_size=args.image_size,
    )
    test_subset = dataset.get_subset("test")
    test_loader = get_eval_loader(
        "standard", test_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Split: {args.split_scheme}  |  Batches: {len(test_loader)}")

    stream_eval = TreePolygonsStreamingEvalState(dataset)

    # Accumulate only enough samples for visualization
    viz_cap: dict[int, int] = {}
    viz_y_pred: List[dict] = []
    viz_y_true: List[dict] = []
    viz_metadata_rows: List[torch.Tensor] = []
    viz_tf_points: List[torch.Tensor] = []  # cyan dots — parallel to viz_y_pred

    images_dir = os.path.join(str(dataset._data_dir), "images")
    use_tile = args.patch_size > 0 or args.patch_fraction > 0

    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        if not use_tile:
            images_tensor = torch.as_tensor(images)
            tf_preds = model.predict_step(images_tensor, b_idx)
        else:
            tf_preds = None  # computed per-image below

        batch_y_pred: List[dict] = []
        batch_y_true: List[dict] = []
        batch_pts: List[torch.Tensor] = []

        for i, (image_meta, image_tensor, target) in enumerate(zip(metadata, images, targets)):
            H, W = image_tensor.shape[1], image_tensor.shape[2]
            img_np = to_numpy_rgb(image_tensor)

            if use_tile:
                filename_id = int(image_meta[0])
                basename = dataset._filename_id_to_code[filename_id]
                img_path = os.path.join(images_dir, basename)

                if args.patch_fraction > 0:
                    with Image.open(img_path) as _im:
                        _w, _h = _im.size
                    patch_size = max(1, int(min(_w, _h) * args.patch_fraction))
                else:
                    patch_size = args.patch_size

                tile_df = model.predict_tile(
                    path=img_path,
                    patch_size=patch_size,
                    patch_overlap=0.0,
                )
                torch.cuda.empty_cache()

                if tile_df is None or len(tile_df) == 0:
                    pts_cpu = torch.zeros((0, 2))
                    points_df = pd.DataFrame(columns=["x", "y", "score", "label"])
                else:
                    # Extract x/y — predict_tile may return x/y columns or point geometry
                    if "x" in tile_df.columns and "y" in tile_df.columns:
                        x_orig = tile_df["x"].astype(float).to_numpy()
                        y_orig = tile_df["y"].astype(float).to_numpy()
                    else:
                        x_orig = tile_df.geometry.x.astype(float).to_numpy()
                        y_orig = tile_df.geometry.y.astype(float).to_numpy()
                    sc = tile_df["score"].astype(float).to_numpy() if "score" in tile_df.columns else np.ones(len(tile_df))

                    # Scale from original image pixel space to eval image_size (H×W)
                    with Image.open(img_path) as orig_img:
                        orig_w, orig_h = orig_img.size
                    x_scaled = x_orig * (W / orig_w)
                    y_scaled = y_orig * (H / orig_h)

                    pts_cpu = torch.tensor(np.column_stack([x_scaled, y_scaled]), dtype=torch.float32)
                    points_df = pd.DataFrame({
                        "x": x_scaled, "y": y_scaled,
                        "score": sc, "label": "Tree",
                    })
            else:
                tf_pred = tf_preds[i]
                raw_points = tf_pred.get("points", torch.zeros((0, 2)))
                pts_cpu = raw_points.detach().cpu() if isinstance(raw_points, torch.Tensor) else torch.zeros((0, 2))
                if len(pts_cpu) > 0:
                    sc = tf_pred["scores"].detach().cpu().numpy()
                    points_df = pd.DataFrame({
                        "x": pts_cpu[:, 0].numpy(),
                        "y": pts_cpu[:, 1].numpy(),
                        "score": sc,
                        "label": "Tree",
                    })
                else:
                    points_df = pd.DataFrame(columns=["x", "y", "score", "label"])

            batch_pts.append(pts_cpu)

            if len(pts_cpu) == 0:
                y_pred_img = _empty_pred(H, W)
            else:
                kw = dict(
                    results=points_df,
                    image=img_np,
                    prompt_mode="point",
                    model_name=args.sam2_model,
                    hf_token=args.hf_token,
                    iou_threshold=args.iou_threshold,
                    mask_threshold=args.mask_threshold,
                    prompt_batch_size=args.prompt_batch_size,
                    use_negative_point_prompts=args.negative_prompts,
                )
                if args.max_point_prompts is not None:
                    kw["max_point_prompts"] = args.max_point_prompts
                poly_gdf = model.predict_polygons(**kw)
                y_pred_img = polygons_to_prediction(poly_gdf, H, W)

            batch_y_pred.append(y_pred_img)
            batch_y_true.append(target)
            batch_pts.append(pts_cpu)

        # Stream metrics update — no need to hold the full dataset in memory
        batch_meta = metadata[:len(batch_y_pred)]
        stream_eval.update(batch_y_pred, batch_y_true, batch_meta)

        # Accumulate viz samples (capped at viz_n_per_source per source)
        if args.viz_dir:
            for j in range(len(batch_y_pred)):
                sid = int(batch_meta[j, 1].item())
                if viz_cap.get(sid, 0) < args.viz_n_per_source:
                    viz_y_pred.append(batch_y_pred[j])
                    viz_y_true.append(batch_y_true[j])
                    viz_metadata_rows.append(batch_meta[j])
                    viz_tf_points.append(batch_pts[j])
                    viz_cap[sid] = viz_cap.get(sid, 0) + 1

        print(f"  batch {b_idx + 1}/{len(test_loader)}", flush=True)

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    results, results_str = stream_eval.finalize()
    print(results_str)

    if args.viz_dir:
        save_viz_with_tf_points(
            dataset=dataset,
            y_pred=viz_y_pred,
            y_true=viz_y_true,
            tf_points=viz_tf_points,
            metadata=torch.stack(viz_metadata_rows) if viz_metadata_rows else torch.zeros((0, 2), dtype=torch.long),
            out_dir=args.viz_dir,
            n_per_source=args.viz_n_per_source,
            score_threshold=args.iou_threshold,
            viz_size=args.viz_size,
        )
        print(f"Visualizations written to {args.viz_dir}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f"results_polygons_{args.split_scheme}.txt"), "w") as f:
            f.write(results_str)
        flat = {}
        for k, v in results.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        flat[f"{k}/{k2}"] = v2
            elif isinstance(v, (int, float)):
                flat[k] = v
        with open(os.path.join(args.output_dir, f"results_polygons_{args.split_scheme}.json"), "w") as f:
            json.dump(
                {"model": "TreeFormer+SAM3", "task": "TreePolygons",
                 "split": args.split_scheme, "metrics": flat},
                f, indent=2,
            )
        print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
