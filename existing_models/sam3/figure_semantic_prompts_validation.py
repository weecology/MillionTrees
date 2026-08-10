"""SAM3 semantic-prompt figure on held-out images.

Runs SAM3 with three text prompts ("tree", "tree canopy", "individual tree
canopy") on a handful of images and overlays the predicted masks with alpha so
we can see how the semantic concept changes what gets segmented. "individual
tree canopy" is the concept the polygon model currently uses.

Produces, per image, a row of panels: ground truth + one panel per prompt.

`--source-name` takes one or more sources, so a single figure can contrast a
source SAM3 handles well against one it struggles on. `--eval-split` takes one
value per source, since Allen/Frey only exist in the validation split:

    --source-name "Allen et al. 2025" "SelvaMask" \
      --eval-split validation test --num-images 1 --output-tag good_vs_bad
"""

import argparse
import os
from typing import List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from milliontrees import get_dataset
from milliontrees.common.metrics.all_metrics import MaskAccuracy

# Prompts, coarse -> specific. Last one is what the polygon model uses today.
PROMPTS = ["tree", "tree canopy", "individual tree canopy"]
# One color per prompt panel (orange, blue, green).
COLORS = [(1.0, 0.55, 0.0), (0.0, 0.7, 1.0), (0.3, 0.85, 0.3)]
GT_COLOR = (0.6, 0.1, 0.8)  # purple


def to_pil(x_float_hwc: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(x_float_hwc, 0, 1) * 255).astype(np.uint8))


def sam3_masks(model, processor, pil_img, prompt, device, post_threshold, mask_threshold):
    inputs = processor(images=[pil_img], text=[prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=post_threshold, mask_threshold=mask_threshold,
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


def overlay(ax, base_img, masks, color, alpha=0.45, edge_width=2,
            edge_color=(0.05, 0.05, 0.05), edge_alpha=0.95):
    """Fill each instance with a translucent color, then outline every instance.

    These are instance masks, not a semantic map, but with dozens of overlapping
    crowns the fills merge into one blob. Outlines go on in a second pass and in a
    dark color so a crown drawn later cannot erase the boundary of one drawn
    earlier — that is the only thing showing where one prediction ends.
    """
    ax.imshow(base_img)
    if len(masks) == 0:
        return
    from scipy import ndimage
    overlay_rgba = np.zeros((*base_img.shape[:2], 4), dtype=np.float32)
    masks_np = [m.numpy().astype(bool) for m in masks]
    for mn in masks_np:
        overlay_rgba[mn, 0] = color[0]
        overlay_rgba[mn, 1] = color[1]
        overlay_rgba[mn, 2] = color[2]
        overlay_rgba[mn, 3] = alpha
    if edge_width > 0:
        for mn in masks_np:
            border = mn & ~ndimage.binary_erosion(mn, iterations=edge_width)
            overlay_rgba[border, :3] = edge_color
            overlay_rgba[border, 3] = edge_alpha
    ax.imshow(overlay_rgba)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", type=str,
                   default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    p.add_argument("--source-name", type=str, nargs="+", default=["Allen et al. 2025"],
                   help="One or more sources to draw images from (one block of rows each).")
    p.add_argument("--split-scheme", type=str, default="out-of-distribution",
                   choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    p.add_argument("--eval-split", type=str, nargs="+", default=["validation"],
                   help="Split to draw from: one value for all sources, or one per source "
                        "(e.g. Allen lives in validation, most sources in test).")
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-images", type=int, default=2,
                   help="How many images to render per source (rows).")
    p.add_argument("--target-gt", type=int, default=25,
                   help="Prefer images whose GT polygon count is nearest this.")
    p.add_argument("--image-indices", type=int, nargs="*", default=None,
                   help="Force specific dataset indices (overrides target-gt selection). "
                        "Only valid with a single --source-name.")
    p.add_argument("--score-threshold", type=float, default=0.1,
                   help="Score threshold for kept predictions (leaderboard standard 0.1).")
    p.add_argument("--post-threshold", type=float, default=0.05)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--iou-threshold", type=float, default=0.4)
    p.add_argument("--alpha", type=float, default=0.35,
                   help="Fill opacity per instance. Kept lowish so the dark outlines read.")
    p.add_argument("--edge-width", type=int, default=2,
                   help="Instance outline thickness in image pixels; 0 disables outlines.")
    p.add_argument("--pred-color", type=str, default=None,
                   help="Single matplotlib color for every prediction panel (e.g. 'orange'). "
                        "Default gives each prompt its own color; a single color makes the "
                        "panels differ only by predicted shape and count.")
    p.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    p.add_argument("--output-dir", type=str,
                   default="/blue/ewhite/b.weinstein/src/MillionTrees/docs/public")
    p.add_argument("--output-tag", type=str, default=None,
                   help="Filename suffix; defaults to the source names joined by '_'.")
    p.add_argument("--formats", type=str, nargs="+", default=["png", "svg"],
                   help="Figure formats to write. SVG keeps the titles as editable text "
                        "(the panels themselves stay embedded raster).")
    p.add_argument("--hspace", type=float, default=0.18,
                   help="Vertical gap between rows, as a fraction of axes height. Needs to "
                        "clear the two-line panel titles.")
    args = p.parse_args()

    device = torch.device(args.device)
    metric = MaskAccuracy(iou_threshold=args.iou_threshold)

    colors = COLORS
    if args.pred_color:
        from matplotlib.colors import to_rgb
        colors = [to_rgb(args.pred_color)] * len(PROMPTS)

    dataset = get_dataset("TreePolygons", root_dir=args.root_dir, download=False,
                          split_scheme=args.split_scheme, image_size=args.image_size)

    name_to_id = {v: k for k, v in dataset._source_id_to_code.items()}
    for name in args.source_name:
        if name not in name_to_id:
            raise SystemExit(f"{name!r} not found. Sources: {list(name_to_id)}")
    if args.image_indices and len(args.source_name) > 1:
        raise SystemExit("--image-indices only works with a single --source-name.")

    splits = args.eval_split
    if len(splits) == 1:
        splits = splits * len(args.source_name)
    elif len(splits) != len(args.source_name):
        raise SystemExit("--eval-split takes one value or one per --source-name.")
    subsets = {s: dataset.get_subset(s) for s in set(splits)}

    # Choose images (deterministic): nearest to target GT count, per source.
    chosen = []  # (source_name, dataset index)
    for name, split in zip(args.source_name, splits):
        subset = subsets[split]
        src_col = subset.metadata_array[:, 1]
        src_id = name_to_id[name]
        cand_positions = torch.nonzero(src_col == src_id, as_tuple=False).squeeze(1).tolist()
        print(f"{name}: {len(cand_positions)} {split} images")
        if not cand_positions:
            raise SystemExit(f"No candidate images for {name!r} in {split}.")
        if args.image_indices:
            chosen += [(name, i) for i in args.image_indices]
            continue
        scored = []
        for pos in cand_positions:
            orig_idx = int(subset.indices[pos])
            _, _, tgt = dataset[orig_idx]
            n_gt = len(tgt["y"])
            if n_gt == 0:
                continue
            scored.append((abs(n_gt - args.target_gt), orig_idx, n_gt))
        scored.sort(key=lambda t: t[0])
        for _, idx, n_gt in scored[:args.num_images]:
            print(f"Selected {name}: dataset idx {idx} with {n_gt} GT polygons")
            chosen.append((name, idx))

    # Load SAM3 once.
    from transformers import Sam3Processor, Sam3Model  # type: ignore
    print("Loading SAM3 ...")
    model = Sam3Model.from_pretrained("facebook/sam3", token=args.hf_token).to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3", token=args.hf_token)

    n_rows = len(chosen)
    n_cols = len(PROMPTS) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.8 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    all_rows = []
    for r, (src_name, orig_idx) in enumerate(chosen):
        metadata, x, target = dataset[orig_idx]
        fname = dataset._input_array[orig_idx]
        pil_img = to_pil(x)
        base_img = np.asarray(pil_img)
        gt_masks = target["y"]
        if not isinstance(gt_masks, torch.Tensor):
            gt_masks = torch.as_tensor(np.asarray(gt_masks))
        gt_masks = gt_masks.bool()
        n_gt = len(gt_masks)
        print(f"\nImage row {r}: {src_name} / {fname}  "
              f"({base_img.shape[1]}x{base_img.shape[0]})  GT polygons: {n_gt}")

        overlay(axes[r, 0], base_img, gt_masks, GT_COLOR, alpha=args.alpha,
                edge_width=args.edge_width)
        axes[r, 0].set_title(f"Ground truth\n{src_name}  n={n_gt}", fontsize=11)

        for i, prompt in enumerate(PROMPTS):
            masks, scores = sam3_masks(model, processor, pil_img, prompt, device,
                                       args.post_threshold, args.mask_threshold)
            keep = scores > args.score_threshold
            masks_k, scores_k = masks[keep], scores[keep]
            if len(masks_k) > 0:
                iou = metric._mask_iou(gt_masks, masks_k)
                recall = float(metric._recall(gt_masks, masks_k, args.iou_threshold, iou=iou))
            else:
                recall = 0.0
            overlay(axes[r, i + 1], base_img, masks_k, colors[i], alpha=args.alpha,
                    edge_width=args.edge_width)
            axes[r, i + 1].set_title(
                f'"{prompt}"\nn_pred={len(masks_k)}  recall={recall:.2f}', fontsize=11)
            all_rows.append((src_name, fname, prompt, len(masks_k), recall))
            print(f"  prompt={prompt!r:24s} n_pred(>{args.score_threshold})={len(masks_k):4d} "
                  f"recall@iou{args.iou_threshold}={recall:.3f}")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(
        f"SAM3 semantic-prompt sensitivity on {'+'.join(dict.fromkeys(splits))} imagery\n"
        f"(score_threshold={args.score_threshold}, mask-IoU recall @ {args.iou_threshold})",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # tight_layout packs square images edge to edge, which leaves each row's two-line
    # titles sitting on top of the row above. Re-open the vertical gap afterwards.
    fig.subplots_adjust(hspace=args.hspace)

    os.makedirs(args.output_dir, exist_ok=True)
    split_tag = "_".join(dict.fromkeys(splits))
    tag = args.output_tag or "_".join(n.split()[0].lower() for n in args.source_name)
    stem = os.path.join(args.output_dir, f"sam3_prompt_sensitivity_{split_tag}_{tag}")
    for fmt in args.formats:
        out_fig = f"{stem}.{fmt}"
        fig.savefig(out_fig, dpi=140, bbox_inches="tight")
        print(f"\nSaved figure -> {out_fig}")

    out_csv = os.path.join(args.output_dir, f"sam3_prompt_sensitivity_{split_tag}_{tag}.csv")
    with open(out_csv, "w") as f:
        f.write("source,filename,prompt,n_pred,recall\n")
        for src_name, fname, prompt, n_pred, recall in all_rows:
            f.write(f'"{src_name}","{fname}","{prompt}",{n_pred},{recall:.4f}\n')
    print(f"Saved metrics -> {out_csv}")


if __name__ == "__main__":
    main()
