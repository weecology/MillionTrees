"""Profile TreePolygons evaluation on the mini (or full) dataset.

Measures per-phase wall time, mask/pair counts, and cProfile hotspots for
legacy ``dataset.eval()`` vs streaming ``TreePolygonsStreamingEvalState``.

Example:
    uv run python scripts/profile_polygon_eval.py --root-dir profiling_data --mini
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader
from milliontrees.datasets.polygon_stream_eval import TreePolygonsStreamingEvalState


@dataclass
class DatasetStats:
    n_images: int = 0
    n_gt_masks: int = 0
    n_pred_masks: int = 0
    gt_masks_per_image: list[int] = field(default_factory=list)
    pred_masks_per_image: list[int] = field(default_factory=list)
    mask_pixels: int = 0

    @property
    def pairwise_gt_pred(self) -> int:
        return sum(g * p for g, p in zip(self.gt_masks_per_image,
                                         self.pred_masks_per_image))

    @property
    def avg_gt_per_image(self) -> float:
        return self.n_gt_masks / max(self.n_images, 1)

    @property
    def avg_pred_per_image(self) -> float:
        return self.n_pred_masks / max(self.n_images, 1)


def _mock_preds_from_targets(targets: list[dict],
                             *,
                             pred_fraction: float = 1.0) -> list[dict]:
    """Use GT masks as predictions (realistic dense crowns for timing)."""
    preds: list[dict] = []
    for target in targets:
        gt_masks = target["y"]
        if not isinstance(gt_masks, torch.Tensor):
            gt_masks = torch.as_tensor(gt_masks)
        gt_masks = gt_masks.to(torch.uint8)
        if gt_masks.ndim == 2:
            gt_masks = gt_masks.unsqueeze(0)
        n = gt_masks.shape[0]
        if n == 0:
            preds.append({
                "y": torch.zeros((0, 448, 448), dtype=torch.uint8),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "scores": torch.zeros((0,), dtype=torch.float32),
            })
            continue
        if pred_fraction <= 1.0:
            keep_n = max(1, int(n * pred_fraction))
            pred_masks = gt_masks[:keep_n]
        else:
            reps = int(pred_fraction)
            pred_masks = gt_masks.repeat(reps, 1, 1)
            keep_n = pred_masks.shape[0]
        scores = torch.linspace(0.95, 0.55, keep_n)
        preds.append({
            "y": pred_masks,
            "labels": torch.zeros((keep_n,), dtype=torch.int64),
            "scores": scores,
        })
    return preds


def _collect_stats(y_pred: list, y_true: list, score_threshold: float) -> DatasetStats:
    stats = DatasetStats()
    for pred, gt in zip(y_pred, y_true):
        scores = pred["scores"]
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores)
        n_pred = int((scores > score_threshold).sum().item())
        gt_masks = gt["y"]
        n_gt = len(gt_masks)
        stats.n_images += 1
        stats.n_gt_masks += n_gt
        stats.n_pred_masks += n_pred
        stats.gt_masks_per_image.append(n_gt)
        stats.pred_masks_per_image.append(n_pred)
        if n_gt > 0:
            stats.mask_pixels += int(gt_masks.shape[-2] * gt_masks.shape[-1] * n_gt)
    return stats


def _timed(label: str, fn, timings: dict[str, float]):
    t0 = time.perf_counter()
    out = fn()
    timings[label] = time.perf_counter() - t0
    return out


def _profile_call(label: str, fn) -> tuple[object, str]:
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - t0
    pr.disable()
    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(25)
    header = f"\n=== cProfile: {label} ({elapsed:.3f}s) ===\n"
    return out, header + buf.getvalue()


def _run_stream_eval(dataset, y_pred, y_true, metadata, timings: dict[str, float]):
    state = TreePolygonsStreamingEvalState(dataset)
    batch_size = 4
    n = len(y_pred)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_pred = y_pred[start:end]
        batch_true = y_true[start:end]
        batch_meta = metadata[start:end]
        _timed(
            "stream_update_batch",
            lambda bp=batch_pred, bt=batch_true, bm=batch_meta: state.update(
                bp, bt, bm),
            timings,
        )

    for key in state._EW_KEYS:
        _timed(
            f"stream_finalize_ew_{key}",
            lambda k=key: state._finalize_elementwise(k, dataset.metrics[k]),
            timings,
        )
    _timed("stream_finalize_map_global", state._map_global.compute, timings)
    gcnt = state._ew["accuracy"]["g_cnt"]
    for gi in range(state._n_groups):
        if gcnt[gi] <= 0:
            continue
        _timed(
            f"stream_finalize_map_group_{gi}",
            lambda g=gi: state._map_per_group[g].compute(),
            timings,
        )
    return state.finalize()


def _run_legacy_eval(dataset, y_pred, y_true, metadata, timings: dict[str, float]):
    map_metric = dataset.metrics["AP50"]
    for metric_name, metric in dataset.metrics.items():
        if metric_name == "counting_mae":
            continue
        if hasattr(metric, "_compute_element_wise"):
            _timed(
                f"legacy_ew_{metric_name}",
                lambda m=metric: m._compute_element_wise(y_pred, y_true),
                timings,
            )

    _timed("legacy_map_format", lambda: map_metric._format(y_pred, y_true), timings)
    preds, targets = map_metric._format(y_pred, y_true)
    from torchmetrics.detection import MeanAveragePrecision

    def _global_map():
        m = MeanAveragePrecision(
            iou_type=map_metric.iou_type,
            iou_thresholds=map_metric.iou_thresholds,
            max_detection_thresholds=map_metric.max_detection_thresholds,
            class_metrics=False,
        )
        m.update(preds, targets)
        return m.compute()

    _timed("legacy_map_compute_global", _global_map, timings)

    g = dataset._eval_grouper.metadata_to_group(metadata)
    for gi in range(dataset._eval_grouper.n_groups):
        mask = g == gi
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0].tolist()
        gp = [y_pred[i] for i in idx]
        gt = [y_true[i] for i in idx]

        def _group_map(gp=gp, gt=gt):
            m = MeanAveragePrecision(
                iou_type=map_metric.iou_type,
                iou_thresholds=map_metric.iou_thresholds,
                max_detection_thresholds=map_metric.max_detection_thresholds,
                class_metrics=False,
            )
            p2, t2 = map_metric._format(gp, gt)
            m.update(p2, t2)
            return m.compute()

        _timed(f"legacy_map_compute_group_{gi}", _group_map, timings)

    return None


def _print_timings(title: str, timings: dict[str, float]) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    total = sum(timings.values())
    for key, sec in sorted(timings.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * sec / total if total else 0.0
        print(f"  {sec:8.3f}s  ({pct:5.1f}%)  {key}")
    print(f"  {'—' * 8}")
    print(f"  {total:8.3f}s  (100.0%)  [sum of labeled phases]")


def _print_stats(stats: DatasetStats, n_groups: int) -> None:
    print(f"\n{'=' * 60}")
    print("Dataset / mask statistics (test split)")
    print(f"{'=' * 60}")
    print(f"  Images:              {stats.n_images}")
    print(f"  Source groups:       {n_groups}")
    print(f"  GT masks (total):    {stats.n_gt_masks}")
    print(f"  Pred masks (total):  {stats.n_pred_masks}")
    print(f"  Avg GT / image:      {stats.avg_gt_per_image:.1f}")
    print(f"  Avg pred / image:    {stats.avg_pred_per_image:.1f}")
    print(f"  Max GT / image:      {max(stats.gt_masks_per_image) if stats.gt_masks_per_image else 0}")
    print(f"  Max pred / image:    {max(stats.pred_masks_per_image) if stats.pred_masks_per_image else 0}")
    print(f"  Pairwise pred×GT:    {stats.pairwise_gt_pred:,}")
    print(f"  Mask tensor pixels:  {stats.mask_pixels:,}  (H×W×N_gt, 448² each)")
    full_scale_images = 1695
    full_scale_factor = full_scale_images / max(stats.n_images, 1)
    est_pairs_full = stats.pairwise_gt_pred * full_scale_factor
    print(f"\n  Extrapolation to ~{full_scale_images} test images (full release):")
    print(f"    Pairwise pred×GT ≈ {est_pairs_full:,.0f}")
    print(f"    MAP compute calls: 1 global + {n_groups} per-source = {n_groups + 1}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=str, default="profiling_data")
    parser.add_argument("--mini", action="store_true", default=True)
    parser.add_argument("--no-mini", action="store_false", dest="mini")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pred-fraction", type=float, default=1.0,
                        help="Fraction of GT crowns kept as predictions (1.0 = dense).")
    parser.add_argument("--profile-map-only", action="store_true",
                        help="Run cProfile only on global MAP compute().")
    args = parser.parse_args()

    dataset = get_dataset(
        "TreePolygons",
        root_dir=args.root_dir,
        mini=args.mini,
        download=False,
        split_scheme="random",
        verbose=True,
    )
    test_subset = dataset.get_subset("test")
    loader = get_eval_loader("standard", test_subset, batch_size=args.batch_size)

    load_t0 = time.perf_counter()
    all_y_true: list[dict] = []
    for _, _, targets in loader:
        all_y_true.extend(targets)
    load_sec = time.perf_counter() - load_t0

    all_y_pred = _mock_preds_from_targets(all_y_true,
                                          pred_fraction=args.pred_fraction)
    metadata = test_subset.metadata_array[:len(all_y_true)]
    stats = _collect_stats(all_y_pred, all_y_true, dataset.eval_score_threshold)

    print(f"\nDataloader: {load_sec:.3f}s for {len(all_y_true)} images")
    _print_stats(stats, int(dataset._eval_grouper.n_groups))

    if args.profile_map_only:
        map_metric = dataset.metrics["AP50"]
        preds, targets = map_metric._format(all_y_pred, all_y_true)
        from torchmetrics.detection import MeanAveragePrecision

        metric = MeanAveragePrecision(
            iou_type=map_metric.iou_type,
            iou_thresholds=map_metric.iou_thresholds,
            max_detection_thresholds=map_metric.max_detection_thresholds,
            class_metrics=False,
        )
        metric.update(preds, targets)
        _, prof = _profile_call("AP50_global_compute", metric.compute)
        print(prof)
        return

    legacy_timings: dict[str, float] = {}
    stream_timings: dict[str, float] = {}

    print("\nProfiling legacy metric phases...")
    _run_legacy_eval(dataset, all_y_pred, all_y_true, metadata, legacy_timings)
    _print_timings("Legacy eval phase timings", legacy_timings)

    print("\nProfiling stream eval...")
    stream_results, stream_prof = _profile_call(
        "stream_eval_finalize",
        lambda: _run_stream_eval(dataset, all_y_pred, all_y_true, metadata,
                                 stream_timings),
    )
    _print_timings("Stream eval phase timings", stream_timings)
    print(stream_prof)

    print(f"\nStream eval summary (first lines):\n{stream_results[1][:400]}")

    map_legacy = legacy_timings.get("legacy_map_compute_global", 0.0)
    map_groups_legacy = sum(
        v for k, v in legacy_timings.items() if k.startswith("legacy_map_compute_group_"))
    map_stream = stream_timings.get("stream_finalize_map_global", 0.0)
    map_groups_stream = sum(
        v for k, v in stream_timings.items()
        if k.startswith("stream_finalize_map_group_"))

    print(f"\n{'=' * 60}")
    print("AP50 compute() summary")
    print(f"{'=' * 60}")
    print(f"  Legacy  global compute:  {map_legacy:.3f}s")
    print(f"  Legacy  per-group sum:   {map_groups_legacy:.3f}s  ({int(dataset._eval_grouper.n_groups)} groups)")
    print(f"  Legacy  MAP total:       {map_legacy + map_groups_legacy:.3f}s")
    print(f"  Stream  global compute:  {map_stream:.3f}s")
    print(f"  Stream  per-group sum:   {map_groups_stream:.3f}s")
    print(f"  Stream  MAP total:       {map_stream + map_groups_stream:.3f}s")

    full_images = 1695
    scale = full_images / max(stats.n_images, 1)
    for label, t in [("Legacy MAP", map_legacy + map_groups_legacy),
                     ("Stream MAP", map_stream + map_groups_stream)]:
        est_hours = (t * scale) / 3600.0
        print(f"\n  Extrapolated {label} on ~{full_images} images: {est_hours:.2f} hours")


if __name__ == "__main__":
    main()
