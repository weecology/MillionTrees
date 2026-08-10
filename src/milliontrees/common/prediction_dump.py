"""Persist raw eval predictions + ground truth so metrics can be recomputed offline.

The leaderboard eval scripts only write aggregate metric summaries, so any new question ("what is AP
at IoU 0.4?", "what is AP on just the completely-annotated tiles?") means re-running the model.
Dumping predictions once decouples inference from scoring: the model runs on the split, this module
writes everything the metrics need, and ``scripts/rescore_validation_ap.py`` re-scores any subset at
any IoU without a GPU.

Everything is stored in the *evaluation frame* (the resized image the model actually saw, e.g.
448x448 for TreeBoxes), which is the frame ``y_pred``/``y_true`` already live in, so no coordinate
mapping is needed. Polygon masks are RLE-encoded (pycocotools) as each batch arrives, so dumping a
polygon run costs one batch of dense masks, not the whole split.

Format: a single ``.pkl`` holding ``{"meta": {...}, "images": [ {...}, ... ]}``.
"""

import os
import pickle

import numpy as np


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def _encode_masks(masks):
    """RLE-encode a [N, H, W] boolean mask stack (pycocotools, Fortran order)."""
    from pycocotools import mask as mask_utils

    masks = _to_numpy(masks)
    if masks is None or masks.ndim != 3 or masks.shape[0] == 0:
        shape = tuple(
            masks.shape[1:]) if masks is not None and masks.ndim == 3 else (0,
                                                                            0)
        return {"rle": [], "shape": shape}
    arr = np.asfortranarray(masks.astype(np.uint8).transpose(1, 2, 0))
    return {"rle": mask_utils.encode(arr), "shape": tuple(masks.shape[1:])}


def decode_masks(packed):
    """Inverse of ``_encode_masks``: returns an [N, H, W] boolean array."""
    from pycocotools import mask as mask_utils

    rle = packed.get("rle") or []
    h, w = packed.get("shape", (1, 1)) or (1, 1)
    if not rle:
        return np.zeros((0, max(int(h), 1), max(int(w), 1)), dtype=bool)
    arr = mask_utils.decode(rle)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr.transpose(2, 0, 1).astype(bool)


def _pack_example(entry, geometry_name, iou_type):
    """Pull geometry / scores / labels out of one prediction or target dict."""
    out = {}
    # Prediction dicts from the eval scripts key geometry as "y"; targets use the
    # dataset's geometry_name. Accept either.
    geo = entry.get(geometry_name)
    if geo is None:
        geo = entry.get("y")
    if iou_type == "segm":
        out["masks"] = _encode_masks(geo)
    else:
        arr = _to_numpy(geo)
        if arr is None or arr.size == 0:
            arr = np.zeros((0, 4), dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 4)
        out["geometry"] = arr
    scores = _to_numpy(entry.get("scores"))
    if scores is not None:
        out["scores"] = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = _to_numpy(entry.get("labels"))
    if labels is not None:
        out["labels"] = np.asarray(labels, dtype=np.int64).reshape(-1)
    if "complete" in entry:
        val = entry["complete"]
        out["complete"] = bool(val.item() if hasattr(val, "item") else val)
    return out


class PredictionDumper:
    """Accumulates RLE/array-encoded predictions batch by batch, writes one .pkl."""

    def __init__(self,
                 path,
                 dataset,
                 model=None,
                 task=None,
                 split_scheme=None,
                 eval_split=None,
                 score_threshold=None):
        self.path = path
        self.geometry_name = getattr(dataset, "geometry_name", "y")
        self.iou_type = "segm" if dataset.dataset_name == "TreePolygons" else "bbox"
        fields = list(getattr(dataset, "metadata_fields", []))
        self.filename_idx = fields.index(
            "filename_id") if "filename_id" in fields else None
        self.source_idx = fields.index(
            "source_id") if "source_id" in fields else None
        self.filename_map = getattr(dataset, "_filename_id_to_code", {}) or {}
        self.source_map = getattr(dataset, "_source_id_to_code", {}) or {}
        self.images = []
        self.meta = {
            "model": model,
            "task": task or dataset.dataset_name,
            "geometry_name": self.geometry_name,
            "iou_type": self.iou_type,
            "split_scheme": split_scheme,
            "eval_split": eval_split,
            "score_threshold": score_threshold,
            "image_size": getattr(dataset, "image_size", None),
        }

    def update(self, y_pred, y_true, metadata):
        metadata = _to_numpy(metadata)
        for i, (pred, true) in enumerate(zip(y_pred, y_true)):
            fid = int(metadata[
                i, self.filename_idx]) if self.filename_idx is not None else -1
            sid = int(metadata[
                i, self.source_idx]) if self.source_idx is not None else -1
            self.images.append({
                "filename": self.filename_map.get(fid),
                "filename_id": fid,
                "source_id": sid,
                "source": self.source_map.get(sid),
                "pred": _pack_example(pred, self.geometry_name, self.iou_type),
                "true": _pack_example(true, self.geometry_name, self.iou_type),
            })

    def close(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({
                "meta": self.meta,
                "images": self.images
            },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(self.images)} predictions to {self.path}")
        return self.path


def add_dump_args(parser):
    """Add ``--save-predictions`` to an eval script's argument parser."""
    parser.add_argument(
        "--save-predictions",
        type=str,
        default=None,
        help="Write raw predictions + ground truth to this .pkl so metrics "
        "(AP at other IoUs, subsets of tiles) can be recomputed offline.")
    return parser


def make_dumper(args, dataset, model=None, task=None):
    """Return a ``PredictionDumper`` if ``--save-predictions`` was passed, else None."""
    path = getattr(args, "save_predictions", None)
    if not path:
        return None
    return PredictionDumper(path,
                            dataset,
                            model=model,
                            task=task,
                            split_scheme=getattr(args, "split_scheme", None),
                            eval_split=getattr(args, "eval_split", None),
                            score_threshold=getattr(args, "score_threshold",
                                                    None))


def maybe_save_predictions(args,
                           dataset,
                           subset,
                           y_pred,
                           y_true,
                           model=None,
                           task=None):
    """One-shot dump for scripts that already hold every prediction in memory."""
    dumper = make_dumper(args, dataset, model=model, task=task)
    if dumper is None:
        return None
    dumper.update(y_pred, y_true, subset.metadata_array[:len(y_pred)])
    return dumper.close()


def load_predictions(path):
    with open(path, "rb") as f:
        return pickle.load(f)
