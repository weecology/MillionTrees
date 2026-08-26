"""Shared Detectron2 pieces for the polygon weak-supervision ablation.

The box ablation (``training/boxes/train.py`` + ``pretrain_backbone_for_polygons.py``)
asked whether pretraining on the 6.4M unsupervised Weinstein et al. 2018 boxes helps a
tree detector. It answered "no" on the DeepForest / torchvision RetinaNet stack. This
module rebuilds the same three-arm comparison on **native Detectron2**, the stack that
actually produces the leaderboard polygon numbers, so the null can be attributed to the
weak labels rather than to DeepForest.

Three things live here because both stages need them:

``load_weak_box_dicts``
    TreeBoxes ``<split>.csv`` -> Detectron2 dataset dicts. Weak rows carry a box and no
    polygon, so each annotation gets its box rectangle as ``segmentation`` (shapes must
    line up for ``annotations_to_instances``) and is flagged ``real_mask=False``.

``MaskAwareROIHeads`` + ``WeakMaskDatasetMapper``
    The co-training arm mixes box-only weak images with polygon images in one batch. The
    flag set by the mapper rides through ``label_and_sample_proposals`` (which copies any
    ``gt_*`` field from targets onto sampled proposals) and the ROI heads drop the
    box-only proposals before the mask loss. Without this the mask head would be trained
    to predict rectangles, which is a different and worse experiment.

``verify_coco_init`` / ``export_transferable`` / ``load_transferable``
    The init hygiene the box round had to learn twice: assert tensor-for-tensor that the
    run really started from COCO (DeepForest's config default silently substituted a
    NEON-trained checkpoint in two earlier ablations), and transfer the *whole* stage-1
    network rather than the trunk alone (the trunk-only arm was worth -0.016 AP40).
"""

import fnmatch
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals
from detectron2.structures import BoxMode

# Everything the stage-1 network can hand to stage 2. The mask head is deliberately
# absent: stage 1 never trains it, so it stays at COCO init in stage 2.
TRANSFERABLE_PREFIXES = ("backbone.", "proposal_generator.", "roi_heads.box_")
# Tensors that must be byte-identical to the COCO release for an "init: coco" claim.
COCO_VERIFY_PREFIXES = ("backbone.",)


# --------------------------------------------------------------------------- #
# TreeBoxes -> Detectron2 bridge (weak, box-only supervision)
# --------------------------------------------------------------------------- #
def _matches_any(name, patterns):
    return any(fnmatch.fnmatch(str(name), pat) for pat in patterns)


def _select_filenames(df, limit_images, stratify_by_source):
    """Pick which images to keep, spreading a cap evenly over sources when asked.

    The supervised holdout that drives checkpoint selection is only a few hundred images
    out of ~13k across 16 sources; taking them in filename order would draw the whole
    holdout from one or two sources and make the recall signal a property of that source
    rather than of the detector.
    """
    per_image = df[["filename", "source"]].drop_duplicates("filename")
    if limit_images is None or len(per_image) <= limit_images:
        return None
    if not stratify_by_source:
        return set(per_image["filename"].iloc[:limit_images])
    buckets = [g["filename"].tolist() for _, g in per_image.groupby("source", sort=True)]
    chosen = []
    idx = 0
    while len(chosen) < limit_images and any(idx < len(b) for b in buckets):
        for bucket in buckets:
            if idx < len(bucket):
                chosen.append(bucket[idx])
                if len(chosen) == limit_images:
                    break
        idx += 1
    return set(chosen)


def load_weak_box_dicts(data_dir, split_scheme, split="train", source_patterns=("*unsupervised*",),
                        exclude_patterns=(), image_id_offset=0, limit_images=None,
                        stratify_by_source=False):
    """Build Detectron2 dataset dicts from the box-only rows of a TreeBoxes split.

    ``source_patterns`` selects which sources count as weak supervision (the default
    picks up ``Weinstein et al. 2018 unsupervised``, the 6.4M-box source stage 1 trains
    on and the one the box co-training arm mixed in). Each record is flagged
    ``real_mask=False`` so the mask head can be held out of the loss downstream; the flag
    is per *image* because a TreeBoxes tile is box-only end to end, which is what lets
    :class:`WeakMaskDatasetMapper` attach it without re-deriving which annotations
    survived augmentation.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / f"{split_scheme}.csv"
    cols = ["filename", "source", "split", "xmin", "ymin", "xmax", "ymax"]
    df = pd.read_csv(csv_path, low_memory=False, usecols=cols)
    df = df[df["split"] == split]
    df = df[df["source"].map(lambda s: _matches_any(s, source_patterns))]
    if exclude_patterns:
        df = df[~df["source"].map(lambda s: _matches_any(s, exclude_patterns))]
    df = df.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    if df.empty:
        raise ValueError(
            f"No weak box rows in {csv_path} for split={split} sources={source_patterns}"
        )

    keep = _select_filenames(df, limit_images, stratify_by_source)
    if keep is not None:
        df = df[df["filename"].isin(keep)]

    images_dir = data_dir / "images"
    records = []
    for filename, group in df.groupby("filename", sort=True):
        image_path = images_dir / str(filename)
        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except FileNotFoundError:
            # Missing imagery is a packaging bug, not something to paper over silently.
            raise
        annotations = []
        for xmin, ymin, xmax, ymax in group[["xmin", "ymin", "xmax", "ymax"]].to_numpy():
            x0 = float(np.clip(xmin, 0, width))
            y0 = float(np.clip(ymin, 0, height))
            x1 = float(np.clip(xmax, 0, width))
            y1 = float(np.clip(ymax, 0, height))
            if x1 <= x0 or y1 <= y0:
                continue
            annotations.append({
                "bbox": [x0, y0, x1, y1],
                "bbox_mode": BoxMode.XYXY_ABS,
                # Rectangle stand-in so annotations_to_instances can build gt_masks;
                # real_mask=False keeps it out of the mask loss.
                "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
                "category_id": 0,
            })
        if not annotations:
            continue
        records.append({
            "file_name": str(image_path),
            "image_id": image_id_offset + len(records),
            "height": height,
            "width": width,
            "annotations": annotations,
            "real_mask": False,
        })
    return records


def mark_real_masks(records, real=True):
    """Tag an existing set of dicts (e.g. the polygon ones) with ``real_mask``."""
    for rec in records:
        rec.setdefault("real_mask", real)
    return records


# --------------------------------------------------------------------------- #
# Mask-loss masking for mixed box/polygon batches
# --------------------------------------------------------------------------- #
class WeakMaskDatasetMapper(DatasetMapper):
    """``DatasetMapper`` that carries the per-image ``real_mask`` flag through.

    ``annotations_to_instances`` only understands the keys it knows about, so the flag is
    re-attached afterwards as ``gt_real_mask``. The ``gt_`` prefix is load-bearing:
    :meth:`ROIHeads.label_and_sample_proposals` copies exactly those fields from the
    ground-truth instances onto the sampled proposals, which is how the flag reaches the
    mask branch. Reading it off the record rather than the annotations means augmentation
    can drop instances freely without the flags falling out of alignment.
    """

    def __call__(self, dataset_dict):
        real_mask = bool(dataset_dict.get("real_mask", True))
        out = super().__call__(dataset_dict)
        instances = out.get("instances")
        if instances is not None:
            instances.gt_real_mask = torch.full((len(instances),), real_mask, dtype=torch.bool)
        return out


@ROI_HEADS_REGISTRY.register()
class MaskAwareROIHeads(StandardROIHeads):
    """StandardROIHeads that trains the mask branch only on real polygon instances."""

    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return {} if self.training else instances
        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = [
                inst[inst.gt_real_mask] if inst.has("gt_real_mask") else inst
                for inst in instances
            ]
        if self.mask_pooler is not None:
            feats = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            feats = self.mask_pooler(feats, boxes)
        else:
            feats = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(feats, instances)


# --------------------------------------------------------------------------- #
# Init verification and whole-network transfer
# --------------------------------------------------------------------------- #
def _load_pkl_state(path):
    with open(path, "rb") as fh:
        blob = pickle.load(fh, encoding="latin1")
    return blob["model"] if isinstance(blob, dict) and "model" in blob else blob


def verify_coco_init(model, coco_pkl, prefixes=COCO_VERIFY_PREFIXES, label="Mask R-CNN"):
    """Assert every backbone/FPN tensor equals the released COCO weights.

    Prints the same one-line receipt the box arms print, so a log can be grepped for
    proof that an arm labelled "coco" really started from COCO.
    """
    coco = _load_pkl_state(coco_pkl)
    checked = 0
    mismatched = []
    for name, tensor in model.state_dict().items():
        if not name.startswith(prefixes):
            continue
        if name not in coco:
            mismatched.append(f"{name} (absent from COCO release)")
            continue
        ref = torch.as_tensor(np.asarray(coco[name]))
        if ref.shape != tensor.shape or not torch.allclose(
                ref.to(tensor.dtype), tensor.detach().cpu(), atol=1e-6):
            mismatched.append(name)
        checked += 1
    if mismatched:
        raise RuntimeError(
            f"COCO init verification FAILED for {len(mismatched)} tensors "
            f"(first few: {mismatched[:5]}). Refusing to run an arm labelled 'coco'."
        )
    print(f"Verified COCO init: all {checked} backbone/FPN tensors match "
          f"the released Detectron2 {label} COCO weights")
    return checked


def assert_not_coco_init(model, coco_pkl, prefixes=COCO_VERIFY_PREFIXES):
    """The mirror image of :func:`verify_coco_init`, for the pretrained arm.

    A merged pkl that silently failed to apply would leave the model at COCO and quietly
    turn the sequential arm into a second copy of the control -- which is exactly the
    failure mode that made the earlier trunk-only arm ambiguous. Assert that the backbone
    actually moved.
    """
    coco = _load_pkl_state(coco_pkl)
    same = differing = 0
    for name, tensor in model.state_dict().items():
        if not name.startswith(prefixes) or name not in coco:
            continue
        ref = torch.as_tensor(np.asarray(coco[name]))
        if ref.shape == tensor.shape and torch.allclose(
                ref.to(tensor.dtype), tensor.detach().cpu(), atol=1e-6):
            same += 1
        else:
            differing += 1
    if differing == 0:
        raise RuntimeError(
            "Backbone is byte-identical to COCO after loading the stage-1 weights: the "
            "pretrained arm did not actually load. Refusing to run a control in disguise."
        )
    print(f"Stage-1 init confirmed: {differing} backbone/FPN tensors differ from COCO "
          f"({same} unchanged)")
    return differing


def export_transferable(model, out_path, prefixes=TRANSFERABLE_PREFIXES):
    """Write the stage-1 tensors stage 2 can consume, as a Detectron2 ``.pkl``."""
    state = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
        if name.startswith(prefixes)
    }
    if not state:
        raise RuntimeError(f"Nothing matched {prefixes}; refusing to write an empty export")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump({"model": state, "__author__": "MillionTrees-stage1",
                     "matching_heuristics": False}, fh)
    print(f"Full network export written to {out_path} ({len(state)} tensors: "
          f"backbone + FPN + RPN + ROI box head)")
    return len(state)


def merge_transferable_into_coco(stage1_pkl, coco_pkl, out_path):
    """Overlay a stage-1 export on the COCO weights so the mask head stays COCO-init.

    Detectron2 loads one ``MODEL.WEIGHTS`` file, so the two have to be merged on disk.
    Every stage-1 tensor must land on an existing COCO key with a matching shape --
    a silent partial load is what made the earlier trunk-only arm hard to interpret.
    """
    stage1 = _load_pkl_state(stage1_pkl)
    merged = dict(_load_pkl_state(coco_pkl))
    replaced, added, reshaped = 0, [], []
    for name, arr in stage1.items():
        arr = np.asarray(arr)
        if name in merged:
            ref = np.asarray(merged[name])
            if ref.shape != arr.shape:
                # Expected for the box predictor only: COCO carries 80 classes, both
                # MillionTrees stages carry one. Stage 1's shape is the correct one.
                reshaped.append(f"{name} {ref.shape}->{arr.shape}")
            replaced += 1
        else:
            added.append(name)
        merged[name] = arr
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump({"model": merged, "__author__": "MillionTrees-stage1-merged",
                     "matching_heuristics": False}, fh)
    mask_keys = sum(1 for k in merged if k.startswith("roi_heads.mask_"))
    print(f"Merged stage-1 into COCO: {replaced} tensors replaced, {len(added)} new "
          f"({added[:4]}{'...' if len(added) > 4 else ''}), "
          f"{len(reshaped)} reshaped ({reshaped}), "
          f"{mask_keys} mask-head tensors left at COCO init -> {out_path}")
    return {"replaced": replaced, "added": added, "reshaped": reshaped,
            "mask_head_keys_at_coco": mask_keys}
