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
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import time
import shapely.affinity
import torch
from PIL import Image, ImageDraw
from rasterio.features import rasterize as rio_rasterize
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape
from shapely.ops import unary_union

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


def polygons_to_prediction_native(poly_gdf, eval_h: int, eval_w: int,
                                   native_h: int, native_w: int) -> dict:
    """Rasterize polygons in native pixel coords, then resize to eval resolution."""
    if poly_gdf is None or len(poly_gdf) == 0:
        return _empty_pred(eval_h, eval_w)

    masks: List[np.ndarray] = []
    scores_list: List[float] = []
    for _, row in poly_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        m_native = rio_rasterize(
            [(geom.__geo_interface__, 1)],
            out_shape=(native_h, native_w),
            fill=0, dtype=np.uint8,
        ).astype(bool)
        m_eval = np.array(
            Image.fromarray(m_native.astype(np.uint8) * 255)
            .resize((eval_w, eval_h), Image.Resampling.NEAREST)
        ) > 127
        masks.append(m_eval)
        score = float(row["score"]) if "score" in poly_gdf.columns and pd.notna(row["score"]) else 1.0
        scores_list.append(score)

    if not masks:
        return _empty_pred(eval_h, eval_w)

    return {
        "y": torch.tensor(np.stack(masks), dtype=torch.bool),
        "labels": torch.zeros(len(masks), dtype=torch.int64),
        "scores": torch.tensor(scores_list, dtype=torch.float32),
    }


def _slug(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(name), flags=re.UNICODE).strip("_")
    return s[:max_len] if len(s) > max_len else s or "source"


def _mask_to_polygon_scaled(mask_np: np.ndarray, eval_h: int, eval_w: int, orig_h: int, orig_w: int):
    """Extract polygon from a boolean mask and scale to original image dimensions."""
    m = mask_np.astype(np.uint8)
    polys = [shape(g) for g, v in rio_shapes(m, mask=m) if v == 1]
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    sx, sy = orig_w / eval_w, orig_h / eval_h
    if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
        merged = shapely.affinity.scale(merged, xfact=sx, yfact=sy, origin=(0, 0))
    return merged


def _polygon_to_svg_path(polygon) -> str:
    """Convert a shapely Polygon or MultiPolygon to an SVG path d attribute."""
    from shapely.geometry import MultiPolygon
    parts = list(polygon.geoms) if isinstance(polygon, MultiPolygon) else [polygon]
    d_parts = []
    for poly in parts:
        coords = list(poly.exterior.coords)
        d = f"M {coords[0][0]:.2f},{coords[0][1]:.2f}"
        for x, y in coords[1:]:
            d += f" L {x:.2f},{y:.2f}"
        d += " Z"
        for interior in poly.interiors:
            ic = list(interior.coords)
            d += f" M {ic[0][0]:.2f},{ic[0][1]:.2f}"
            for x, y in ic[1:]:
                d += f" L {x:.2f},{y:.2f}"
            d += " Z"
        d_parts.append(d)
    return " ".join(d_parts)


def save_full_scale_viz_and_svg(
    img_path: str,
    y_pred: dict,
    y_true: dict,
    tf_points,
    eval_size: int,
    out_stem: Path,
    score_threshold: float = 0.5,
) -> None:
    """Save a full-resolution PNG and an editable SVG for one image.

    SVG groups:
      #ground-truth   — purple GT polygon fills
      #predictions    — orange predicted polygon fills
      #treeformer-points — cyan tree centroid circles
    """
    with Image.open(img_path).convert("RGB") as orig_img:
        orig_w, orig_h = orig_img.size
        orig_np = np.array(orig_img)

    gt_m = _tensor_masks(y_true.get("y"), eval_size)
    pred_m = _tensor_masks(y_pred.get("y"), eval_size)
    scores = y_pred.get("scores")
    if scores is not None and pred_m.shape[0] > 0:
        sc = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores, dtype=torch.float32)
        pred_m = pred_m[sc.detach().cpu().float().numpy() > score_threshold]

    sx, sy = orig_w / eval_size, orig_h / eval_size

    # --- Full-resolution PNG ---
    arr = orig_np.copy().astype(np.float32)
    for mask_set, color in [(gt_m, _COLOR_GROUND_TRUTH), (pred_m, _COLOR_PREDICTION)]:
        for idx in range(mask_set.shape[0]):
            m_eval = (mask_set[idx].numpy() if isinstance(mask_set[idx], torch.Tensor) else mask_set[idx]).astype(np.uint8)
            m_orig = np.array(
                Image.fromarray(m_eval * 255).resize((orig_w, orig_h), Image.Resampling.NEAREST)
            ) > 127
            for c in range(3):
                arr[:, :, c][m_orig] = arr[:, :, c][m_orig] * 0.65 + float(color[c]) * 0.35
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    png_img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(png_img)
    if tf_points is not None and len(tf_points) > 0:
        pts_np = tf_points.detach().cpu().float().numpy() if isinstance(tf_points, torch.Tensor) else np.asarray(tf_points)
        r_circ = max(6, int(6 * min(sx, sy)))
        for px, py in pts_np:
            px_s, py_s = float(px) * sx, float(py) * sy
            draw.ellipse([px_s - r_circ, py_s - r_circ, px_s + r_circ, py_s + r_circ],
                         outline=_COLOR_TF_POINTS, width=2)
    png_img.save(str(out_stem) + ".png")

    # --- SVG ---
    buf = BytesIO()
    Image.fromarray(orig_np).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    def _mask_group(mask_set, color, gid):
        r, g, b = color
        paths = []
        for idx in range(mask_set.shape[0]):
            m_np = mask_set[idx].numpy() if isinstance(mask_set[idx], torch.Tensor) else mask_set[idx]
            poly = _mask_to_polygon_scaled(m_np, eval_size, eval_size, orig_h, orig_w)
            if poly is not None:
                paths.append(f'    <path d="{_polygon_to_svg_path(poly)}" />')
        if not paths:
            return []
        return (
            [f'  <g id="{gid}" fill="rgba({r},{g},{b},0.35)" stroke="rgb({r},{g},{b})"'
             f' stroke-width="{max(1, int(min(sx, sy)))}" fill-rule="evenodd">']
            + paths
            + ["  </g>"]
        )

    svg = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"'
        f' width="{orig_w}" height="{orig_h}" viewBox="0 0 {orig_w} {orig_h}">',
        f'  <image href="data:image/png;base64,{img_b64}" x="0" y="0" width="{orig_w}" height="{orig_h}"/>',
    ]
    svg += _mask_group(gt_m, _COLOR_GROUND_TRUTH, "ground-truth")
    svg += _mask_group(pred_m, _COLOR_PREDICTION, "predictions")

    if tf_points is not None and len(tf_points) > 0:
        pts_np = tf_points.detach().cpu().float().numpy() if isinstance(tf_points, torch.Tensor) else np.asarray(tf_points)
        r_val, g_val, b_val = _COLOR_TF_POINTS
        r_circ = max(5, int(5 * min(sx, sy)))
        svg.append(f'  <g id="treeformer-points" fill="rgba({r_val},{g_val},{b_val},0.8)"'
                   f' stroke="rgb({r_val},{g_val},{b_val})" stroke-width="1.5">')
        for px, py in pts_np:
            svg.append(f'    <circle cx="{float(px) * sx:.1f}" cy="{float(py) * sy:.1f}" r="{r_circ}" />')
        svg.append("  </g>")

    svg.append("</svg>")
    with open(str(out_stem) + ".svg", "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def save_viz_with_tf_points(
    dataset,
    y_pred: List[dict],
    y_true: List[dict],
    tf_points: List[torch.Tensor],
    metadata: torch.Tensor,
    out_dir: str,
    n_per_source: int | None = 10,
    score_threshold: float = 0.5,
    viz_size: int | None = None,
    img_paths: Optional[List[str]] = None,
) -> None:
    """Write overlay PNGs (and SVGs when img_paths provided) per source.

    When img_paths is given, also calls save_full_scale_viz_and_svg for each
    sample, producing full-resolution PNGs and editable SVGs alongside the
    standard downscaled PNGs.
    """
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
        if n_per_source is not None and per_source_count.get(source_id, 0) >= n_per_source:
            continue

        filename_id = int(metadata[row, 0].item())
        ds_idx = fid_to_ds_idx[filename_id]

        raw = _to_numpy_image(dataset.get_input(ds_idx))
        rgb = _resize_eval_image(raw, eval_size)
        gt_m = _tensor_masks(y_true[row].get("y"), eval_size)
        pm = _tensor_masks(y_pred[row].get("y"), eval_size)
        scores = y_pred[row].get("scores")
        if scores is not None and pm.shape[0] > 0:
            sc = scores if isinstance(scores, torch.Tensor) else torch.as_tensor(scores, dtype=torch.float32)
            pm = pm[sc.detach().cpu().float().numpy() > score_threshold]

        arr = _blend_masks(rgb, gt_m, _COLOR_GROUND_TRUTH, alpha=0.35)
        arr = _blend_masks(arr, pm, _COLOR_PREDICTION, alpha=0.35)

        base = Image.fromarray(arr, mode="RGB")
        if image_size != eval_size:
            base = base.resize((image_size, image_size), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(base)

        pts = tf_points[row]
        if pts is not None and len(pts) > 0:
            pts_np = pts.detach().cpu().float().numpy() if isinstance(pts, torch.Tensor) else np.asarray(pts)
            scale = image_size / eval_size
            pts_np = pts_np * scale
            _draw_points(draw, pts_np, _COLOR_TF_POINTS, r=max(4, int(4 * scale)))

        src_name = getattr(dataset, "_source_id_to_code", {}).get(source_id, str(source_id))
        stem = Path(getattr(dataset, "_filename_id_to_code", {}).get(filename_id, f"id_{filename_id}")).stem
        sub = out_path / _slug(src_name)
        sub.mkdir(parents=True, exist_ok=True)
        k = per_source_count.get(source_id, 0)
        fname_stem = sub / f"{k:03d}_{_slug(stem)}"
        base.save(str(fname_stem) + "_thumb.png")

        # Full-scale PNG + SVG when original image path is available
        if img_paths is not None and row < len(img_paths) and os.path.exists(img_paths[row]):
            save_full_scale_viz_and_svg(
                img_path=img_paths[row],
                y_pred=y_pred[row],
                y_true=y_true[row],
                tf_points=pts,
                eval_size=eval_size,
                out_stem=fname_stem,
                score_threshold=score_threshold,
            )

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
        choices=["crossgeometry", "random", "zeroshot"],
        help="crossgeometry is the headline benchmark; random/zeroshot "
             "evaluate the same pipeline against the standard TreePolygons test splits.",
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
    parser.add_argument("--profile", action="store_true",
                        help="Print per-batch timing breakdown (TreeFormer vs SAM2 vs data load)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--viz-dir", type=str, default=None,
                        help="Directory for per-source overlay PNGs "
                             "(purple=GT, orange=SAM3 masks, cyan=TreeFormer points)")
    parser.add_argument("--viz-n-per-source", type=lambda x: None if x.lower() == "none" else int(x), default=10,
                        help="Max images per source for viz (default 10); pass 'none' for all")
    parser.add_argument("--viz-size", type=int, default=512,
                        help="Output PNG resolution (pixels); independent of --image-size used for eval")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--checkpoint", type=str,
                        default=TREEFORMER_CONFIG["model"]["name"],
                        help="TreeFormer weights: HuggingFace repo ID or local checkpoint dir")
    parser.add_argument("--revision", type=str,
                        default=TREEFORMER_CONFIG["model"]["revision"],
                        help="Model revision/tag (ignored for local checkpoint dirs)")
    args = parser.parse_args()

    # Visualization on by default: 10 overlays per source (--viz-n-per-source).
    if args.viz_dir is None:
        args.viz_dir = os.path.join(args.output_dir, "viz") if args.output_dir else "eval_viz"
    elif args.viz_dir == "":
        args.viz_dir = None

    from deepforest import main as df_main

    config_args = dict(TREEFORMER_CONFIG)
    config_args["score_thresh"] = args.score_thresh_tf
    config_args["model"] = dict(TREEFORMER_CONFIG["model"])
    config_args["model"]["name"] = args.checkpoint
    config_args["model"]["revision"] = args.revision
    config_args["point"] = dict(TREEFORMER_CONFIG["point"])
    config_args["point"]["nms_distance_thresh"] = args.nms_distance_thresh
    print(f"TreeFormer checkpoint: {args.checkpoint} (revision={args.revision})")
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
    viz_tf_points: List[torch.Tensor] = []
    viz_img_paths: List[str] = []  # original image paths for full-scale PNG+SVG

    images_dir = os.path.join(str(dataset._data_dir), "images")
    use_tile = args.patch_size > 0 or args.patch_fraction > 0

    # Profiling state — only used when --profile is set
    _p_t_tf = _p_t_sam2 = _p_t_data = 0.0
    _p_n_img = _p_n_batches = _p_n_pts = 0
    _p_t_loop_start = _p_t_batch_end = time.perf_counter()

    for b_idx, (metadata, images, targets) in enumerate(test_loader):
        if args.profile and b_idx > 0:
            _p_t_data += time.perf_counter() - _p_t_batch_end

        if not use_tile:
            images_tensor = torch.as_tensor(images)
            if args.profile:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _tf_t0 = time.perf_counter()
            tf_preds = model.predict_step(images_tensor, b_idx)
            if args.profile:
                if torch.cuda.is_available(): torch.cuda.synchronize()
                _p_t_tf += time.perf_counter() - _tf_t0
        else:
            tf_preds = None  # computed per-image in the tile branch

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

                if args.profile:
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    _tile_t0 = time.perf_counter()
                tile_df = model.predict_tile(
                    path=img_path,
                    patch_size=patch_size,
                    patch_overlap=0.0,
                )
                torch.cuda.empty_cache()
                if args.profile:
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    _p_t_tf += time.perf_counter() - _tile_t0

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
                if args.profile:
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    _sam_t0 = time.perf_counter()
                    _p_n_pts += len(pts_cpu)
                poly_gdf = model.predict_polygons(**kw)
                if args.profile:
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    _p_t_sam2 += time.perf_counter() - _sam_t0
                y_pred_img = polygons_to_prediction(poly_gdf, H, W)

            batch_y_pred.append(y_pred_img)
            batch_y_true.append(target)

        # Single cache flush after all per-image SAM2 calls in this batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stream metrics update — no need to hold the full dataset in memory
        batch_meta = metadata[:len(batch_y_pred)]
        stream_eval.update(batch_y_pred, batch_y_true, batch_meta)

        # Accumulate viz samples (capped at viz_n_per_source per source)
        if args.viz_dir:
            for j in range(len(batch_y_pred)):
                sid = int(batch_meta[j, 1].item())
                if args.viz_n_per_source is None or viz_cap.get(sid, 0) < args.viz_n_per_source:
                    fid = int(batch_meta[j, 0].item())
                    basename = dataset._filename_id_to_code.get(fid, f"id_{fid}")
                    viz_img_paths.append(os.path.join(images_dir, basename))
                    viz_y_pred.append(batch_y_pred[j])
                    viz_y_true.append(batch_y_true[j])
                    viz_metadata_rows.append(batch_meta[j])
                    viz_tf_points.append(batch_pts[j])
                    viz_cap[sid] = viz_cap.get(sid, 0) + 1

        if args.profile:
            if torch.cuda.is_available(): torch.cuda.synchronize()
            _p_t_batch_end = time.perf_counter()
            _p_n_batches += 1
            _p_n_img += len(batch_y_pred)
            _avg_tf = _p_t_tf / _p_n_batches
            _avg_sam2 = _p_t_sam2 / max(_p_n_img, 1)
            _avg_pts = _p_n_pts / max(_p_n_img, 1)
            _thr = _p_n_img / max(_p_t_batch_end - _p_t_loop_start, 1e-6)
            print(
                f"[profile] batch {b_idx+1}/{len(test_loader)} | "
                f"tf={_avg_tf:.2f}s/batch | "
                f"sam2={_avg_sam2:.2f}s/img ({_avg_pts:.1f}pts/img avg) | "
                f"throughput={_thr:.3f}img/s",
                flush=True,
            )

        print(f"  batch {b_idx + 1}/{len(test_loader)}", flush=True)

        if args.max_batches is not None and (b_idx + 1) >= args.max_batches:
            break

    if args.profile and _p_n_batches > 0:
        _total = time.perf_counter() - _p_t_loop_start
        print(
            f"\n[profile] === Summary over {_p_n_batches} batches, {_p_n_img} images ===\n"
            f"  Data load:  {_p_t_data / max(_p_n_batches-1, 1):.3f}s/batch (between-batch gap)\n"
            f"  TreeFormer: {_p_t_tf / _p_n_batches:.3f}s/batch  ({_p_t_tf / max(_p_n_img, 1):.3f}s/img)\n"
            f"  SAM2:       {_p_t_sam2 / _p_n_batches:.3f}s/batch  "
            f"({_p_t_sam2 / max(_p_n_img, 1):.3f}s/img, {_p_n_pts / max(_p_n_img, 1):.1f}pts/img avg)\n"
            f"  Total wall: {_total:.1f}s  ({_p_n_img / max(_total, 1e-6):.3f}img/s throughput)\n",
            flush=True,
        )

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
            img_paths=viz_img_paths if viz_img_paths else None,
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
                 "split": args.split_scheme, "checkpoint": args.checkpoint,
                 "metrics": flat},
                f, indent=2,
            )
        print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
