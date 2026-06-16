"""Visualize TreeFormer's raw density map *before* peak detection.

For each sampled test image we render a 3-panel figure:
  1. RGB image with ground-truth points (cyan).
  2. The raw 1/4-resolution density map (upsampled), the model's actual output
     before peak_local_max.
  3. Density map overlaid on the image with the detected peaks marked, split
     into peaks the metric keeps (score > eval_score_threshold, lime) vs.
     peaks it drops (red x).

This makes it possible to see whether close-together trees even form separate
modes in the density map, or whether they are merged before peak detection ever
runs. See docs/treeformer_hyperparameters.md.

Example:
    uv run --group treeformer python existing_models/treeformer/visualize_density.py \
        --checkpoint weecology/deepforest-tree-point --split-scheme random \
        --n-per-source 5 --score-integration-radius 2 --score-thresh 0.1 \
        --output-dir existing_models/treeformer/outputs/density_viz
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepforest import main as df_main

from milliontrees import get_dataset
from milliontrees.common.data_loaders import get_eval_loader


def load_model(checkpoint, revision, score_thresh, score_integration_radius):
    if checkpoint.endswith(".ckpt") and os.path.isfile(checkpoint):
        print(f"Loading Lightning checkpoint: {checkpoint}")
        model = df_main.deepforest.load_from_checkpoint(checkpoint, weights_only=False)
    else:
        print(f"Loading TreeFormer weights: {checkpoint} (revision={revision})")
        model = df_main.deepforest(config_args={
            "architecture": "treeformer",
            "model": {"name": checkpoint, "revision": revision},
        })
        model.load_model(checkpoint, revision=revision)
    # Set on the submodule (config is ignored after load_model / load_from_checkpoint).
    model.model.score_thresh = score_thresh
    model.model.score_integration_radius = score_integration_radius
    print(f"score_thresh={model.model.score_thresh} "
          f"score_integration_radius={model.model.score_integration_radius} "
          f"downsample_ratio={getattr(model.model, 'downsample_ratio', '?')} "
          f"density_sigma={getattr(model.model, 'density_sigma', '?')}")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def predict_with_density(model, images):
    """Run inference, capturing the per-image raw density maps via the
    submodule's postprocess_density hook. Returns (preds, density_maps)."""
    captured = {}
    submodule = model.model
    orig = submodule.postprocess_density

    def hook(density_map, imgs):
        # density_map: list of (1,1,H,W) per image (the eval path builds a list).
        captured["density"] = [d.detach().float().cpu() for d in density_map]
        return orig(density_map, imgs)

    submodule.postprocess_density = hook
    try:
        device = next(model.parameters()).device
        imgs = images if isinstance(images, torch.Tensor) else torch.tensor(images)
        with torch.no_grad():
            preds = model.predict_step(imgs.to(device), 0)
    finally:
        submodule.postprocess_density = orig
    return preds, captured.get("density", [])


def render(image_chw, gt_xy, density_hw, pred_xy, pred_scores,
           eval_score_threshold, title, out_path):
    img = np.transpose(image_chw, (1, 2, 0))
    img = np.clip(img, 0, 1)
    H, W = img.shape[:2]

    # Upsample density map to image size for overlay.
    dm = density_hw
    dm_up = np.array(
        torch.nn.functional.interpolate(
            torch.from_numpy(dm)[None, None].float(),
            size=(H, W), mode="bilinear", align_corners=False,
        )[0, 0]
    )

    keep = pred_scores > eval_score_threshold
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(img)
    if len(gt_xy):
        ax[0].scatter(gt_xy[:, 0], gt_xy[:, 1], s=18, c="cyan",
                      edgecolors="black", linewidths=0.4, label=f"GT ({len(gt_xy)})")
    ax[0].set_title("Image + GT points")
    ax[0].legend(loc="upper right", fontsize=8)

    im1 = ax[1].imshow(dm_up, cmap="magma")
    ax[1].set_title(f"Raw density map ({dm.shape[0]}x{dm.shape[1]} -> {H}x{W})")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(img)
    ax[2].imshow(dm_up, cmap="magma", alpha=0.5)
    if keep.any():
        ax[2].scatter(pred_xy[keep, 0], pred_xy[keep, 1], s=24, marker="o",
                      facecolors="none", edgecolors="lime", linewidths=1.2,
                      label=f"kept >{eval_score_threshold:g} ({int(keep.sum())})")
    drop = ~keep
    if drop.any():
        ax[2].scatter(pred_xy[drop, 0], pred_xy[drop, 1], s=24, marker="x",
                      c="red", linewidths=1.0,
                      label=f"dropped ({int(drop.sum())})")
    ax[2].set_title("Peaks on density overlay")
    ax[2].legend(loc="upper right", fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Visualize TreeFormer density maps before peak detection.")
    p.add_argument("--checkpoint", type=str, default="weecology/deepforest-tree-point")
    p.add_argument("--revision", type=str, default="main")
    p.add_argument("--root-dir", type=str,
                   default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    p.add_argument("--split-scheme", type=str, default="random",
                   choices=["random", "zeroshot", "crossgeometry"])
    p.add_argument("--score-thresh", type=float, default=0.1)
    p.add_argument("--score-integration-radius", type=int, default=2)
    p.add_argument("--n-per-source", type=int, default=5)
    p.add_argument("--image-size", type=int, default=448,
                   help="Resize images (and GT points) to this square size before "
                        "inference. The density map is image_size/4 per side, so "
                        "larger sizes give finer separability between close trees.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=str,
                   default="existing_models/treeformer/outputs/density_viz")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.checkpoint, args.revision,
                       args.score_thresh, args.score_integration_radius)

    dataset = get_dataset("TreePoints", root_dir=args.root_dir,
                          split_scheme=args.split_scheme,
                          image_size=args.image_size)
    eval_score_threshold = dataset.eval_score_threshold
    test_subset = dataset.get_subset("test")
    loader = get_eval_loader("standard", test_subset,
                             batch_size=args.batch_size, num_workers=args.num_workers)

    # Source ids that actually appear in the test split, so we can stop early
    # once every one of them has n_per_source examples.
    test_source_ids = set(int(s) for s in test_subset.metadata_array[:, 1].tolist())

    per_source_count = {}
    n_saved = 0
    for metadata, images, targets in loader:
        preds, density_maps = predict_with_density(model, images)
        if not density_maps:
            continue
        for i in range(len(targets)):
            source_id = int(metadata[i, 1])
            source_name = dataset._source_id_to_code.get(source_id, str(source_id))
            if per_source_count.get(source_id, 0) >= args.n_per_source:
                continue
            per_source_count[source_id] = per_source_count.get(source_id, 0) + 1

            gt_xy = targets[i]["y"].cpu().numpy().reshape(-1, 2)
            dm = density_maps[i][0, 0].numpy()
            pred_xy = preds[i]["points"].detach().cpu().numpy().reshape(-1, 2)
            pred_scores = preds[i]["scores"].detach().cpu().numpy().reshape(-1)

            safe = "".join(c if c.isalnum() else "_" for c in source_name)
            out_path = os.path.join(
                args.output_dir, f"{safe}_{per_source_count[source_id]:02d}.png")
            title = (f"{source_name}  |  GT={len(gt_xy)}  peaks={len(pred_xy)}  "
                     f"kept(>{eval_score_threshold:g})={int((pred_scores>eval_score_threshold).sum())}  "
                     f"|  st={args.score_thresh} r={args.score_integration_radius}")
            render(images[i].cpu().numpy(), gt_xy, dm, pred_xy, pred_scores,
                   eval_score_threshold, title, out_path)
            n_saved += 1

        if all(per_source_count.get(s, 0) >= args.n_per_source
               for s in test_source_ids):
            break

    print(f"Saved {n_saved} density visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
