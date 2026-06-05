"""Streaming TreePolygons evaluation without accumulating all preds/GT in memory."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from milliontrees.common.eval_visualization import save_eval_visualizations
from milliontrees.common.metrics.all_metrics import (
    DetectionMAP,
    compute_polygon_mask_elementwise_batch,
    make_mean_average_precision,
)
from milliontrees.common.utils import maximum, minimum


def _disable_torchmetric_sync(metric: Any) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        metric._to_sync = False


class TreePolygonsStreamingEvalState:
    """Accumulates metrics batch-wise; results match ``standard_group_eval`` semantics."""

    _EW_KEYS = ("accuracy", "recall", "maskaware_precision", "merge_commission")
    _MAP_KEY = "AP50"

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset
        self._grouper = dataset._eval_grouper
        self._n_groups = int(self._grouper.n_groups)

        self._ew: dict[str, dict[str, Any]] = {}
        for key in self._EW_KEYS:
            self._ew[key] = {
                "sum": 0.0,
                "n": 0,
                "g_sum": torch.zeros(self._n_groups, dtype=torch.float64),
                "g_cnt": torch.zeros(self._n_groups, dtype=torch.float64),
            }

        self._map_metric: DetectionMAP = dataset.metrics[self._MAP_KEY]
        map_kwargs = dict(
            iou_type=self._map_metric.iou_type,
            iou_thresholds=self._map_metric.iou_thresholds,
            max_detection_thresholds=self._map_metric.max_detection_thresholds,
            class_metrics=False,
        )
        self._map_global = make_mean_average_precision(**map_kwargs)
        _disable_torchmetric_sync(self._map_global)
        self._map_per_group = [
            make_mean_average_precision(**map_kwargs)
            for _ in range(self._n_groups)
        ]
        for m in self._map_per_group:
            _disable_torchmetric_sync(m)

    def update(
        self,
        y_pred: list,
        y_true: list,
        metadata: torch.Tensor,
    ) -> None:
        if not isinstance(metadata, torch.Tensor):
            metadata = torch.as_tensor(metadata)
        g = self._grouper.metadata_to_group(metadata)

        ew_scores = compute_polygon_mask_elementwise_batch(
            y_pred,
            y_true,
            accuracy_metric=self._dataset.metrics["accuracy"],
            recall_metric=self._dataset.metrics["recall"],
            maskaware_metric=self._dataset.metrics["maskaware_precision"],
            merge_metric=self._dataset.metrics["merge_commission"],
        )
        for key in self._EW_KEYS:
            v = ew_scores[key].float()
            if v.device != g.device:
                v = v.to(g.device)
            st = self._ew[key]
            st["sum"] += float(v.sum().item())
            st["n"] += int(v.numel())
            for gi in range(self._n_groups):
                mask = g == gi
                if mask.any():
                    st["g_sum"][gi] += float(v[mask].sum().item())
                    st["g_cnt"][gi] += float(mask.sum().item())

        preds, targets = self._map_metric._format(y_pred, y_true)
        self._map_global.update(preds, targets)

        for gi in range(self._n_groups):
            mask = g == gi
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0].tolist()
            gp = [y_pred[i] for i in idx]
            gt = [y_true[i] for i in idx]
            p2, t2 = self._map_metric._format(gp, gt)
            self._map_per_group[gi].update(p2, t2)

    def _finalize_elementwise(self, key: str, metric: Any) -> tuple[dict, str]:
        st = self._ew[key]
        results: dict[str, Any] = {}
        results_str = ""

        if st["n"] == 0:
            agg = torch.tensor(0.0)
        else:
            agg = torch.tensor(st["sum"] / st["n"])
        results[metric.agg_metric_field] = float(agg.item())
        results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"

        group_avgs = torch.full((self._n_groups,),
                                float("nan"),
                                dtype=torch.float64)
        group_counts = st["g_cnt"]
        for gi in range(self._n_groups):
            if st["g_cnt"][gi] > 0:
                group_avgs[gi] = st["g_sum"][gi] / st["g_cnt"][gi]

        valid = group_counts > 0
        if valid.any():
            sub = group_avgs[valid].float()
            sub = sub[~torch.isnan(sub)]
            if sub.numel() == 0:
                worst = torch.tensor(0.0)
            else:
                worst = metric.worst(sub)
        else:
            worst = torch.tensor(0.0)

        for group_idx in range(self._n_groups):
            group_str = self._grouper.group_field_str(group_idx)
            gv = group_avgs[group_idx]
            results[f"{metric.name}_{group_str}"] = (float(
                gv.item()) if not torch.isnan(gv) else float("nan"))
            results[f"count_{group_str}"] = float(st["g_cnt"][group_idx].item())
            if st["g_cnt"][group_idx] == 0:
                continue
            results_str += (
                f"  {self._grouper.group_str(group_idx)}  "
                f"[n = {int(st['g_cnt'][group_idx].item()):6d}]:\t"
                f"{metric.name} = {float(group_avgs[group_idx].item()):5.3f}\n")

        results[metric.worst_group_metric_field] = float(worst.item())
        results_str += (
            f"Worst-group {metric.name}: {results[metric.worst_group_metric_field]:.3f}\n"
        )
        return results, results_str

    def _finalize_map(self, metric: DetectionMAP) -> tuple[dict, str]:
        results: dict[str, Any] = {}
        results_str = ""

        n_any = int(self._ew["accuracy"]["g_cnt"].sum().item())
        if n_any == 0:
            results[metric.agg_metric_field] = 0.0
            results_str += f"Average {metric.name}: {results[metric.agg_metric_field]:.3f}\n"
            for group_idx in range(self._n_groups):
                group_str = self._grouper.group_field_str(group_idx)
                results[f"{metric.name}_{group_str}"] = 0.0
                results[f"count_{group_str}"] = 0.0
            results[metric.worst_group_metric_field] = 0.0
            results_str += f"Worst-group {metric.name}: 0.000\n"
            return results, results_str

        _disable_torchmetric_sync(self._map_global)
        agg_map = float(self._map_global.compute()["map"].item())
        results[metric.agg_metric_field] = agg_map
        results_str += f"Average {metric.name}: {agg_map:.3f}\n"

        group_metrics_list: list[torch.Tensor] = []
        gcnt = self._ew["accuracy"]["g_cnt"]
        for group_idx in range(self._n_groups):
            group_str = self._grouper.group_field_str(group_idx)
            if gcnt[group_idx] <= 0:
                gv = torch.tensor(0.0)
            else:
                _disable_torchmetric_sync(self._map_per_group[group_idx])
                gv = self._map_per_group[group_idx].compute()["map"]
            group_metrics_list.append(gv)
            results[f"{metric.name}_{group_str}"] = float(gv.item())
            results[f"count_{group_str}"] = float(gcnt[group_idx].item())
            if gcnt[group_idx] == 0:
                continue
            results_str += (f"  {self._grouper.group_str(group_idx)}  "
                            f"[n = {int(gcnt[group_idx].item()):6d}]:\t"
                            f"{metric.name} = {float(gv.item()):5.3f}\n")

        stacked = torch.stack([t.float() for t in group_metrics_list])
        worst = metric.worst(stacked[gcnt > 0])
        results[metric.worst_group_metric_field] = float(worst.item())
        results_str += (
            f"Worst-group {metric.name}: {results[metric.worst_group_metric_field]:.3f}\n"
        )
        return results, results_str

    def finalize(
        self,
        *,
        viz_dir: str | None = None,
        viz_y_pred: list | None = None,
        viz_y_true: list | None = None,
        viz_metadata: torch.Tensor | None = None,
        viz_n_per_source: int | None = 10,
    ) -> tuple[dict[str, Any], str]:
        results: dict[str, Any] = {}
        results_str = ""

        for key in self._EW_KEYS:
            metric = self._dataset.metrics[key]
            r, s = self._finalize_elementwise(key, metric)
            results[key] = r
            results_str += s

        map_metric: DetectionMAP = self._dataset.metrics[self._MAP_KEY]
        r, s = self._finalize_map(map_metric)
        results[self._MAP_KEY] = r
        results_str += s

        # Match ``TreePolygonsDataset.eval`` / ``TreeBoxesDataset.eval`` (same key prefixes).
        detection_accs: list[float] = []
        for k, v in results["accuracy"].items():
            if k.startswith("detection_acc_source:"):
                d = k.split(":")[1]
                count = results["accuracy"].get(f"source:{d}")
                if count and count > 0:
                    detection_accs.append(float(v))
        detection_acc_avg_dom = float(np.array(detection_accs).mean())
        results["detection_acc_avg_dom"] = detection_acc_avg_dom
        results_str = (
            f"Average detection_acc across source: {detection_acc_avg_dom:.3f}\n"
            + results_str)

        from milliontrees.common.utils import format_eval_results

        formatted_results = format_eval_results(results, self._dataset)
        results_str = formatted_results + "\n" + results_str

        if viz_dir is not None and viz_y_pred and viz_y_true and viz_metadata is not None:
            paths = save_eval_visualizations(
                self._dataset,
                viz_y_pred,
                viz_y_true,
                viz_metadata,
                viz_dir,
                n_per_source=viz_n_per_source,
                score_threshold=self._dataset.eval_score_threshold,
            )
            results["eval_visualization_paths"] = [str(p) for p in paths]

        return results, results_str


def merge_viz_samples(
    cap: dict[int, int],
    metadata: torch.Tensor,
    preds: list,
    targets: list,
    *,
    viz_y_pred: list,
    viz_y_true: list,
    viz_rows: list[torch.Tensor],
    n_per_source: int | None,
) -> None:
    """Append up to ``n_per_source`` samples per source_id for visualization."""
    if not isinstance(metadata, torch.Tensor):
        metadata = torch.as_tensor(metadata)
    for i in range(len(preds)):
        sid = int(metadata[i, 1].item())
        if n_per_source is not None and cap.get(sid, 0) >= n_per_source:
            continue
        viz_y_pred.append(preds[i])
        viz_y_true.append(targets[i])
        viz_rows.append(metadata[i].clone())
        cap[sid] = cap.get(sid, 0) + 1
