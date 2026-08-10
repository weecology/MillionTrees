"""Merge an unsupervised box-pretrained ResNet trunk into a COCO Detectron2 pkl.

Table 6 (weak-supervision pretraining) compares a Mask R-CNN whose ResNet
backbone was pretrained on unsupervised tree *boxes* against the generic
COCO/ImageNet backbone. ``training/polygons/train_detectron2.py`` is the native
Detectron2 driver that more than doubled DeepForest's polygon AP50, so the
faithful box-pretrained arm is built here: take the stock COCO Mask R-CNN
weights (``model_final_f10217.pkl`` -- backbone + FPN + RPN + ROI heads all
COCO-pretrained) and overwrite *only* the ResNet bottom-up trunk with the
box-pretrained weights. That mirrors the DeepForest arm, which keeps the
torchvision-COCO FPN/heads and swaps in the box backbone body.

The box backbone (``box_backbone_<split>.pt``) is a torchvision ResNet-50 trunk
saved under ``model.backbone.body.*`` (``IntermediateLayerGetter`` naming).
Detectron2's ResNet uses ``backbone.bottom_up.{stem,res2..res5}.*`` with BatchNorm
folded into each conv as ``.norm``. This script applies the standard
torchvision->D2 key rename (the same mapping as detectron2's
``tools/convert-torchvision-to-d2.py``) and asserts every one of the 265 trunk
tensors lands on an existing COCO key with a matching shape -- so a silent
partial load is impossible.
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import torch

_BODY_PREFIX = "model.backbone.body."
_D2_TRUNK_PREFIX = "backbone.bottom_up."


def torchvision_key_to_d2(key: str) -> str:
    """Map one ``model.backbone.body.*`` torchvision key to its Detectron2 name."""
    assert key.startswith(_BODY_PREFIX), key
    rest = key[len(_BODY_PREFIX):]

    # Stem: conv1 + bn1 -> stem.conv1(.norm)
    if rest.startswith("conv1."):
        return _D2_TRUNK_PREFIX + "stem.conv1." + rest[len("conv1."):]
    if rest.startswith("bn1."):
        return _D2_TRUNK_PREFIX + "stem.conv1.norm." + rest[len("bn1."):]

    # Bottleneck blocks: layer{L}.{B}.<tail> -> res{L+1}.{B}.<tail'>
    m = re.match(r"layer(\d)\.(\d+)\.(.+)", rest)
    if not m:
        raise KeyError(f"unrecognized backbone key: {key}")
    stage, block, tail = int(m.group(1)), m.group(2), m.group(3)
    d2_block = f"{_D2_TRUNK_PREFIX}res{stage + 1}.{block}."

    # conv{n}.weight stays; bn{n}.* -> conv{n}.norm.*
    cm = re.match(r"conv(\d)\.(.+)", tail)
    if cm:
        return d2_block + f"conv{cm.group(1)}.{cm.group(2)}"
    bm = re.match(r"bn(\d)\.(.+)", tail)
    if bm:
        return d2_block + f"conv{bm.group(1)}.norm.{bm.group(2)}"
    # downsample.0 (conv) -> shortcut; downsample.1 (bn) -> shortcut.norm
    dm = re.match(r"downsample\.0\.(.+)", tail)
    if dm:
        return d2_block + f"shortcut.{dm.group(1)}"
    dm = re.match(r"downsample\.1\.(.+)", tail)
    if dm:
        return d2_block + f"shortcut.norm.{dm.group(1)}"
    raise KeyError(f"unrecognized backbone tail: {key}")


def merge(box_backbone_pt: str, coco_pkl: str, out_pkl: str) -> None:
    box_sd = torch.load(box_backbone_pt, map_location="cpu")
    if "state_dict" in box_sd:
        box_sd = box_sd["state_dict"]
    trunk = {k: v for k, v in box_sd.items() if k.startswith(_BODY_PREFIX)}
    if len(trunk) != len(box_sd):
        raise ValueError(
            f"expected all box-backbone keys under {_BODY_PREFIX!r}; got "
            f"{len(trunk)}/{len(box_sd)}"
        )

    with open(coco_pkl, "rb") as f:
        coco = pickle.load(f, encoding="latin1")
    model = coco["model"]  # OrderedDict of numpy arrays

    n_over = 0
    for k, tensor in trunk.items():
        d2_key = torchvision_key_to_d2(k)
        if d2_key not in model:
            raise KeyError(f"{k} -> {d2_key} not in COCO pkl")
        want = tuple(model[d2_key].shape)
        got = tuple(tensor.shape)
        if want != got:
            raise ValueError(f"shape mismatch {d2_key}: coco {want} vs box {got}")
        model[d2_key] = tensor.numpy().astype(np.float32)
        n_over += 1

    if n_over != 265:
        raise ValueError(f"expected to overwrite 265 trunk tensors, did {n_over}")

    out = {"model": model, "__author__": "milliontrees-box-pretrained"}
    Path(out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(out, f)
    print(f"[convert] overwrote {n_over} ResNet-trunk tensors; wrote {out_pkl}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--box-backbone", required=True,
                    help="box_backbone_<split>.pt from pretrain_backbone_for_polygons.py")
    ap.add_argument("--coco-pkl",
                    default="training/polygons/detectron2_assets/model_final_f10217.pkl")
    ap.add_argument("--out", required=True, help="output merged .pkl for --weights")
    args = ap.parse_args()
    merge(args.box_backbone, args.coco_pkl, args.out)


if __name__ == "__main__":
    main()
