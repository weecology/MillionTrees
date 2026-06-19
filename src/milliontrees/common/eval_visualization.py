"""Save qualitative evaluation images (ground truth vs predictions) per source."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# GT vs prediction overlay colors (RGB)
_COLOR_GROUND_TRUTH = (128, 60, 200)  # purple
_COLOR_PREDICTION = (255, 140, 0)  # orange


def _slug(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(name), flags=re.UNICODE).strip("_")
    return s[:max_len] if len(s) > max_len else s or "source"


def _filename_id_to_index(dataset) -> dict[int, int]:
    ma = dataset.metadata_array
    if not isinstance(ma, torch.Tensor):
        ma = torch.as_tensor(ma)
    return {int(ma[i, 0].item()): i for i in range(ma.shape[0])}


def _to_numpy_image(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[0] < x.shape[-1]:
        x = np.transpose(x, (1, 2, 0))
    return np.clip(x.astype(np.float32), 0.0, 1.0)


def _resize_eval_image(img_hwc: np.ndarray, size: int) -> np.ndarray:
    u8 = (np.clip(img_hwc, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")
    pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _tensor_boxes(geo, name: str) -> np.ndarray:
    if geo is None:
        return np.zeros((0, 4), dtype=np.float32)
    t = geo if isinstance(geo, torch.Tensor) else torch.as_tensor(
        geo, dtype=torch.float32)
    t = t.detach().cpu().float()
    if t.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if t.dim() == 1:
        t = t.view(-1, 4)
    return t.numpy()


def _tensor_points(geo) -> np.ndarray:
    t = geo if isinstance(geo, torch.Tensor) else torch.as_tensor(
        geo, dtype=torch.float32)
    t = t.detach().cpu().float()
    if t.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if t.dim() == 1:
        t = t.view(-1, 2)
    return t.numpy()


def _tensor_masks(geo, empty_hw: int) -> np.ndarray:
    if geo is None:
        return np.zeros((0, empty_hw, empty_hw), dtype=np.uint8)
    t = geo if isinstance(geo, torch.Tensor) else torch.as_tensor(geo)
    t = t.detach().cpu()
    if t.numel() == 0:
        if t.ndim == 3 and t.shape[1] > 0 and t.shape[2] > 0:
            return np.zeros((0, int(t.shape[1]), int(t.shape[2])),
                            dtype=np.uint8)
        return np.zeros((0, empty_hw, empty_hw), dtype=np.uint8)
    arr = t.numpy()
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return (arr > 0).astype(np.uint8)


def _draw_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: np.ndarray,
    color: tuple[int, int, int],
    width: int = 2,
) -> None:
    for b in boxes:
        x1, y1, x2, y2 = [float(x) for x in b]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _draw_points(
    draw: ImageDraw.ImageDraw,
    pts: np.ndarray,
    color: tuple[int, int, int],
    r: int = 4,
) -> None:
    for p in pts:
        x, y = float(p[0]), float(p[1])
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)


def _blend_masks(
    base_rgb: np.ndarray,
    masks: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    out = base_rgb.astype(np.float32)
    for i in range(masks.shape[0]):
        m = masks[i].astype(bool)
        if not m.any():
            continue
        for c in range(3):
            ch = out[:, :, c]
            ch[m] = ch[m] * (1.0 - alpha) + float(color[c]) * alpha
            out[:, :, c] = ch
    return np.clip(out, 0, 255).astype(np.uint8)


def save_eval_visualizations(
    dataset,
    y_pred: list[dict],
    y_true: list[dict],
    metadata: torch.Tensor,
    out_dir: str | Path,
    *,
    n_per_source: int | None = 10,
    score_threshold: float | None = None,
) -> list[Path]:
    """Write up to ``n_per_source`` images per ``source_id`` with GT and predictions overlaid.

    Ground truth is drawn in purple; predictions above ``score_threshold`` are orange.

    Images are resized to ``dataset.image_size`` so coordinates match ``y_pred`` / ``y_true`` from
    the standard eval loader.

    Pass ``n_per_source=None`` to write all images without a per-source cap.

    Returns paths of written PNG files.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if len(y_pred) != len(y_true) or len(y_pred) != len(metadata):
        raise ValueError(
            "y_pred, y_true, and metadata must have the same length "
            f"({len(y_pred)}, {len(y_true)}, {len(metadata)}).")

    if not isinstance(metadata, torch.Tensor):
        metadata = torch.as_tensor(metadata)

    fid_to_idx = _filename_id_to_index(dataset)
    geom = dataset.geometry_name
    task = dataset._dataset_name
    image_size = int(getattr(dataset, "image_size", 448))
    if score_threshold is None:
        score_threshold = float(getattr(dataset, "eval_score_threshold", 0.1))

    per_source_count: dict[int, int] = {}
    written: list[Path] = []

    for row in range(len(y_pred)):
        source_id = int(metadata[row, 1].item())
        if n_per_source is not None and per_source_count.get(source_id,
                                                             0) >= n_per_source:
            continue

        filename_id = int(metadata[row, 0].item())
        if filename_id not in fid_to_idx:
            raise KeyError(
                f"filename_id {filename_id} not found in dataset metadata.")
        ds_idx = fid_to_idx[filename_id]

        raw = _to_numpy_image(dataset.get_input(ds_idx))
        rgb = _resize_eval_image(raw, image_size)
        base = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(base)

        gt = y_true[row]
        pr = y_pred[row]

        if task == "TreeBoxes":
            gt_boxes = _tensor_boxes(gt.get(geom), geom)
            scores = pr.get("scores")
            if scores is None:
                pred_boxes = _tensor_boxes(pr.get(geom), geom)
            else:
                sc = scores if isinstance(scores,
                                          torch.Tensor) else torch.as_tensor(
                                              scores, dtype=torch.float32)
                sc = sc.detach().cpu().float()
                geo = pr.get(geom)
                pb = _tensor_boxes(geo, geom)
                if len(pb) == 0:
                    pred_boxes = pb
                else:
                    keep = sc.numpy() > score_threshold
                    pred_boxes = pb[keep]
            _draw_boxes(draw, gt_boxes, _COLOR_GROUND_TRUTH, width=3)
            _draw_boxes(draw, pred_boxes, _COLOR_PREDICTION, width=2)

        elif task == "TreePoints":
            gt_pts = _tensor_points(gt.get(geom))
            scores = pr.get("scores")
            geo = pr.get(geom)
            pp = _tensor_points(geo) if geo is not None else np.zeros(
                (0, 2), dtype=np.float32)
            if scores is None or len(pp) == 0:
                pred_pts = pp
            else:
                sc = scores if isinstance(scores,
                                          torch.Tensor) else torch.as_tensor(
                                              scores, dtype=torch.float32)
                sc = sc.detach().cpu().float().numpy()
                keep = sc > score_threshold
                pred_pts = pp[keep]
            _draw_points(draw, gt_pts, _COLOR_GROUND_TRUTH, r=5)
            _draw_points(draw, pred_pts, _COLOR_PREDICTION, r=4)

        elif task == "TreePolygons":
            gt_m = _tensor_masks(gt.get(geom), image_size)
            scores = pr.get("scores")
            geo = pr.get(geom)
            pm = _tensor_masks(geo, image_size)
            if scores is not None and pm.shape[0] > 0:
                sc = scores if isinstance(scores,
                                          torch.Tensor) else torch.as_tensor(
                                              scores, dtype=torch.float32)
                sc = sc.detach().cpu().float().numpy()
                pm = pm[sc > score_threshold]
            arr = np.asarray(base, dtype=np.uint8)
            arr = _blend_masks(arr, gt_m, _COLOR_GROUND_TRUTH, alpha=0.35)
            arr = _blend_masks(arr, pm, _COLOR_PREDICTION, alpha=0.35)
            base = Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError(
                f"Unsupported dataset for eval visualization: {task}")

        src_name = getattr(dataset, "_source_id_to_code",
                           {}).get(source_id, str(source_id))
        stem = Path(
            getattr(dataset, "_filename_id_to_code",
                    {}).get(filename_id, f"id_{filename_id}")).stem
        sub = out_path / _slug(src_name)
        sub.mkdir(parents=True, exist_ok=True)
        k = per_source_count.get(source_id, 0)
        fname = sub / f"{k:03d}_{_slug(stem)}.png"
        base.save(fname)
        written.append(fname)
        per_source_count[source_id] = k + 1

    return written


def save_count_scatter(counting_summary, out_path, *, title=None):
    """Write a predicted-vs-true count scatter for the points counting eval.

    ``counting_summary`` is the dict produced by ``TreePoints.eval`` under the ``counting_summary``
    key: per-image ``pairs`` (gt / pred / source_id) plus macro ``counting_{nmae,r2,slope}_avg_dom``
    summary scalars. Each point is one complete-annotated image, coloured by source, on log-log axes
    (counts span ~2 orders of magnitude). The 1:1 line marks perfect counting; points sagging below
    it are systematic under-counting. Returns the written path, or None if there is nothing to plot
    (e.g. no complete sources).
    """
    pairs = counting_summary.get("pairs", {})
    gt = np.asarray(pairs.get("gt", []), dtype=float)
    pred = np.asarray(pairs.get("pred", []), dtype=float)
    if gt.size == 0:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_ids = np.asarray(pairs.get("source_id", []), dtype=int)
    per_source = counting_summary.get("per_source", {})

    fig, ax = plt.subplots(figsize=(6, 6))
    # log axes can't show zero counts; nudge to 0.5 so empty tiles stay visible.
    gt_p = np.where(gt <= 0, 0.5, gt)
    pred_p = np.where(pred <= 0, 0.5, pred)

    for sid in sorted(set(source_ids.tolist())):
        m = source_ids == sid
        label = per_source.get(sid, {}).get("source", str(sid))
        ax.scatter(gt_p[m],
                   pred_p[m],
                   s=14,
                   alpha=0.5,
                   label=label,
                   edgecolors="none")

    hi = max(gt_p.max(), pred_p.max()) * 1.2
    lo = 0.4
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("True count (per image)")
    ax.set_ylabel("Predicted count (per image)")

    nmae = counting_summary.get("counting_nmae_avg_dom", float("nan"))
    r2 = counting_summary.get("counting_r2_avg_dom", float("nan"))
    slope = counting_summary.get("counting_slope_avg_dom", float("nan"))
    ax.set_title(title or "Predicted vs. true tree counts")
    ax.text(0.04,
            0.96, f"macro (per-source)\nnMAE = {nmae:.3f}\nR$^2$ = {r2:.3f}\n"
            f"slope = {slope:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
