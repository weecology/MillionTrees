"""Greedy one-to-one matching for detection-style evaluation (IoU or distance).

Used in place of torchvision ``Matcher`` so recall/precision match the usual
GT-centric definitions (aligned with DeepForest-style eval).
"""

from __future__ import annotations

import torch


def greedy_iou_match(iou: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Greedy 1:1 match on an IoU matrix ``[n_gt, n_pred]``.

    Pairs with ``iou > iou_threshold`` are considered, ordered by IoU descending.
    Each ground-truth and each prediction is used at most once.

    Returns:
        LongTensor of shape ``(n_gt,)`` with the matched prediction index, or ``-1``.
    """
    n_gt, n_pred = iou.shape
    device = iou.device
    gt_to_pred = torch.full((n_gt,), -1, dtype=torch.long, device=device)
    if n_gt == 0 or n_pred == 0:
        return gt_to_pred
    valid = iou > iou_threshold
    if not valid.any():
        return gt_to_pred
    scores = iou[valid]
    ij = torch.nonzero(valid, as_tuple=False)
    order = scores.argsort(descending=True)
    pred_used = torch.zeros(n_pred, dtype=torch.bool, device=device)
    for k in order.tolist():
        gi, pj = int(ij[k, 0]), int(ij[k, 1])
        if gt_to_pred[gi] >= 0 or pred_used[pj]:
            continue
        gt_to_pred[gi] = pj
        pred_used[pj] = True
    return gt_to_pred


def greedy_distance_match(dist: torch.Tensor,
                          max_distance: float) -> torch.Tensor:
    """Greedy 1:1 match on a distance matrix ``[n_gt, n_pred]``.

    Pairs with ``dist <= max_distance`` are considered, ordered by distance
    ascending (best spatial alignment first).
    """
    n_gt, n_pred = dist.shape
    device = dist.device
    gt_to_pred = torch.full((n_gt,), -1, dtype=torch.long, device=device)
    if n_gt == 0 or n_pred == 0:
        return gt_to_pred
    valid = dist <= max_distance
    if not valid.any():
        return gt_to_pred
    scores = dist[valid]
    ij = torch.nonzero(valid, as_tuple=False)
    order = scores.argsort(descending=False)
    pred_used = torch.zeros(n_pred, dtype=torch.bool, device=device)
    for k in order.tolist():
        gi, pj = int(ij[k, 0]), int(ij[k, 1])
        if gt_to_pred[gi] >= 0 or pred_used[pj]:
            continue
        gt_to_pred[gi] = pj
        pred_used[pj] = True
    return gt_to_pred


def n_matched_gt(gt_to_pred: torch.Tensor) -> torch.Tensor:
    return (gt_to_pred >= 0).sum().to(dtype=torch.float32)


def merge_commission_rate_iou(iou: torch.Tensor,
                              iou_threshold: float) -> torch.Tensor:
    """Fraction of predictions that exceed ``iou_threshold`` with 2+ GT boxes."""
    n_pred = iou.shape[1]
    if n_pred == 0:
        return torch.tensor(0.0, device=iou.device, dtype=torch.float32)
    hits = (iou > iou_threshold).sum(dim=0)
    n_merge = (hits >= 2).sum().to(dtype=torch.float32)
    return n_merge / float(n_pred)


def merge_commission_rate_distance(dist: torch.Tensor,
                                   max_distance: float) -> torch.Tensor:
    """Fraction of predictions within ``max_distance`` of 2+ GT points."""
    n_pred = dist.shape[1]
    if n_pred == 0:
        return torch.tensor(0.0, device=dist.device, dtype=torch.float32)
    hits = (dist <= max_distance).sum(dim=0)
    n_merge = (hits >= 2).sum().to(dtype=torch.float32)
    return n_merge / float(n_pred)
