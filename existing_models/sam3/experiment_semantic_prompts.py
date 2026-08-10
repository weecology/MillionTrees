"""SAM3 semantic-prompt sensitivity experiment on a single BAMFOREST test image.

BAMFOREST == "Troles et al. 2024" in TreePolygons. We run SAM3 with three text
prompts ("individual tree canopy", "tree", "canopy") on one test image, overlay
the predicted masks against ground truth, and report score recall (greedy mask-IoU
matching at iou=0.4) for each prompt so we can see how sensitive SAM3 is to the
semantic concept used as the prompt.
"""

import argparse
import os
from typing import List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from milliontrees import get_dataset
from milliontrees.common.metrics.all_metrics import MaskAccuracy

SOURCE_NAME = "Troles et al. 2024"  # BAMFOREST
PROMPTS = ["individual tree canopy", "tree", "canopy"]
COLORS = [(1.0, 0.55, 0.0), (0.0, 0.7, 1.0), (0.3, 1.0, 0.3)]  # one per prompt panel


def to_pil(x_float_hwc: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(x_float_hwc, 0, 1) * 255).astype(np.uint8))


def sam3_masks(model, processor, pil_img, prompt, device, score_threshold, mask_threshold):
    inputs = processor(images=[pil_img], text=[prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=score_threshold, mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results.get("masks", None)
    scores = results.get("scores", None)
    if masks is None or len(masks) == 0:
        return (torch.zeros((0, pil_img.height, pil_img.width), dtype=torch.bool),
                torch.zeros((0,), dtype=torch.float32))
    masks_t = torch.as_tensor(masks, dtype=torch.uint8)
    if masks_t.dim() == 2:
        masks_t = masks_t.unsqueeze(0)
    if masks_t.dim() == 4 and masks_t.shape[1] == 1:
        masks_t = masks_t[:, 0]
    masks_t = masks_t.bool().cpu()
    scores_t = (torch.as_tensor(scores, dtype=torch.float32).cpu()
                if scores is not None else torch.zeros(masks_t.shape[0]))
    return masks_t, scores_t


def overlay(ax, base_img, masks, color, alpha=0.45):
    ax.imshow(base_img)
    if len(masks) == 0:
        return
    overlay_rgba = np.zeros((*base_img.shape[:2], 4), dtype=np.float32)
    for m in masks:
        mn = m.numpy().astype(bool)
        overlay_rgba[mn, 0] = color[0]
        overlay_rgba[mn, 1] = color[1]
        overlay_rgba[mn, 2] = color[2]
        overlay_rgba[mn, 3] = alpha
    ax.imshow(overlay_rgba)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=str,
                   default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    p.add_argument("--split-scheme", type=str, default="within-distribution",
                   choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--score-threshold", type=float, default=0.1,
                   help="Score threshold for recall counting (leaderboard standard 0.1).")
    p.add_argument("--post-threshold", type=float, default=0.05,
                   help="SAM3 post_process threshold; keep low, filter for recall separately.")
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.4)
    p.add_argument("--target-gt", type=int, default=30,
                   help="Pick the BAMFOREST test image whose GT count is nearest this.")
    p.add_argument("--image-index", type=int, default=None,
                   help="Force a specific dataset index (overrides target-gt selection).")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    p.add_argument("--output-dir", type=str,
                   default="/blue/ewhite/b.weinstein/src/MillionTrees/docs/public")
    args = p.parse_args()

    device = torch.device(args.device)
    metric = MaskAccuracy(iou_threshold=args.iou_threshold)

    dataset = get_dataset("TreePolygons", root_dir=args.root_dir, download=False,
                          split_scheme=args.split_scheme, image_size=args.image_size)

    name_to_id = {v: k for k, v in dataset._source_id_to_code.items()}
    if SOURCE_NAME not in name_to_id:
        raise SystemExit(f"{SOURCE_NAME} not found. Sources: {list(name_to_id)}")
    src_id = name_to_id[SOURCE_NAME]

    test = dataset.get_subset("test")
    src_col = test.metadata_array[:, 1]
    cand_positions = torch.nonzero(src_col == src_id, as_tuple=False).squeeze(1).tolist()
    print(f"{SOURCE_NAME}: {len(cand_positions)} test images")

    # Choose the image (deterministic). Scan candidates for GT counts.
    if args.image_index is not None:
        orig_idx = args.image_index
    else:
        best = None
        for pos in cand_positions:
            orig_idx = int(test.indices[pos])
            _, _, tgt = dataset[orig_idx]
            n_gt = len(tgt["y"])
            if n_gt == 0:
                continue
            score = abs(n_gt - args.target_gt)
            if best is None or score < best[0]:
                best = (score, orig_idx, n_gt)
            if score == 0:
                break
        orig_idx = best[1]
        print(f"Selected dataset idx {orig_idx} with {best[2]} GT polygons "
              f"(target {args.target_gt})")

    metadata, x, target = dataset[orig_idx]
    fname = dataset._input_array[orig_idx]
    pil_img = to_pil(x)
    base_img = np.asarray(pil_img)

    gt_masks = target["y"]
    if not isinstance(gt_masks, torch.Tensor):
        gt_masks = torch.as_tensor(np.asarray(gt_masks))
    gt_masks = gt_masks.bool()
    n_gt = len(gt_masks)
    print(f"Image: {fname}  ({base_img.shape[1]}x{base_img.shape[0]})  GT polygons: {n_gt}")

    # Load SAM3 once.
    from transformers import Sam3Processor, Sam3Model  # type: ignore
    model = Sam3Model.from_pretrained("facebook/sam3", token=args.hf_token).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=args.hf_token)

    rows = []
    panel_data = []
    for prompt in PROMPTS:
        masks, scores = sam3_masks(model, processor, pil_img, prompt, device,
                                   args.post_threshold, args.mask_threshold)
        keep = scores > args.score_threshold
        masks_k, scores_k = masks[keep], scores[keep]
        y_pred = [{"y": masks_k, "labels": torch.zeros(len(masks_k), dtype=torch.int64),
                   "scores": scores_k}]
        y_true = [{"y": gt_masks, "labels": torch.zeros(n_gt, dtype=torch.int64)}]
        if len(masks_k) > 0:
            iou = metric._mask_iou(gt_masks, masks_k)
            recall = float(metric._recall(gt_masks, masks_k, args.iou_threshold, iou=iou))
        else:
            recall = 0.0
        mean_score = float(scores_k.mean()) if len(scores_k) else 0.0
        rows.append((prompt, len(masks_k), recall, mean_score))
        panel_data.append((prompt, masks_k))
        print(f"prompt={prompt!r:26s} n_pred(>{args.score_threshold})={len(masks_k):4d} "
              f"recall@iou{args.iou_threshold}={recall:.3f} mean_score={mean_score:.3f}")

    # Figure: GT + one panel per prompt.
    fig, axes = plt.subplots(1, len(PROMPTS) + 1, figsize=(5 * (len(PROMPTS) + 1), 5.4))
    overlay(axes[0], base_img, gt_masks, (0.6, 0.1, 0.8))
    axes[0].set_title(f"Ground truth\nBAMFOREST  n={n_gt}", fontsize=11)
    for i, (prompt, masks_k) in enumerate(panel_data):
        ax = axes[i + 1]
        overlay(ax, base_img, masks_k, COLORS[i])
        _, n_pred, recall, _ = rows[i]
        ax.set_title(f'prompt = "{prompt}"\nrecall={recall:.2f}  n_pred={n_pred}', fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"SAM3 semantic-prompt sensitivity — BAMFOREST test image {fname}\n"
        f"(score_threshold={args.score_threshold}, mask-IoU recall @ {args.iou_threshold})",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, "sam3_prompt_sensitivity_bamforest.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure -> {out_png}")

    out_csv = os.path.join(args.output_dir, "sam3_prompt_sensitivity_bamforest.csv")
    with open(out_csv, "w") as f:
        f.write("prompt,n_pred,recall,mean_score\n")
        for prompt, n_pred, recall, mean_score in rows:
            f.write(f'"{prompt}",{n_pred},{recall:.4f},{mean_score:.4f}\n')
    print(f"Saved metrics -> {out_csv}")


if __name__ == "__main__":
    main()
