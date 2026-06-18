"""Composite TreeFormer hyperparameter panel for a single source.

Holds ONE test image fixed and sweeps it across a grid of inference
hyperparameters so the density map and the resulting points can be compared
side by side. This is the figure for the manuscript discussion of how
TreeFormer point detection responds to its knobs.

Layout (one figure per source):
  rows    = image_size           (density-map resolution = image_size / 4)
  columns = [ RGB + GT points,
              raw density map,
              peaks @ radius_1,
              peaks @ radius_2, ... ]

The raw density map depends only on image_size; the peak columns show how
score_integration_radius (peak_local_max min_distance) thins/keeps peaks over
that IDENTICAL map. Peaks kept by the eval score filter are lime, dropped red.

density_sigma (the Gaussian target width) is a TRAIN-time parameter and cannot
be varied at inference on a fixed checkpoint, so it is not a column here.

Example:
    uv run python density_panel.py \
        --source "OSBS megaplot 2025" --source "Ventura et al. 2022" \
        --image-sizes 448 672 896 1280 --radii 1 2 \
        --output-dir outputs/density_panels
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


def load_model(checkpoint, revision, score_thresh):
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
    model.model.score_thresh = score_thresh
    print(f"score_thresh={model.model.score_thresh} "
          f"downsample_ratio={getattr(model.model, 'downsample_ratio', '?')} "
          f"density_sigma={getattr(model.model, 'density_sigma', '?')}")
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def predict_with_density(model, image_chw):
    """Run inference on one (C,H,W) image, capturing the raw density map via the
    submodule's postprocess_density hook. Returns (pred, density_hw)."""
    captured = {}
    submodule = model.model
    orig = submodule.postprocess_density

    def hook(density_map, imgs):
        captured["density"] = [d.detach().float().cpu() for d in density_map]
        return orig(density_map, imgs)

    submodule.postprocess_density = hook
    try:
        device = next(model.parameters()).device
        batch = image_chw.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model.predict_step(batch, 0)
    finally:
        submodule.postprocess_density = orig
    density = captured.get("density", [])
    return preds[0], (density[0][0, 0].numpy() if density else None)


def find_source_id(dataset, source):
    """Resolve a source name/substring to its numeric id."""
    code_to_id = {v: k for k, v in dataset._source_id_to_code.items()}
    if source in code_to_id:
        return code_to_id[source], source
    matches = [(name, sid) for name, sid in code_to_id.items()
               if source.lower() in name.lower()]
    if not matches:
        raise SystemExit(
            f"Source '{source}' not found. Available: {sorted(code_to_id)}")
    if len(matches) > 1:
        raise SystemExit(f"Source '{source}' is ambiguous: {[m[0] for m in matches]}")
    return matches[0][1], matches[0][0]


def pick_densest_index(dataset, source_id, n_candidates):
    """Among the first n_candidates test images of this source, return the
    subset index of the one with the most GT points (densest canopy best shows
    the resolution effect)."""
    test_subset = dataset.get_subset("test")
    source_ids = test_subset.metadata_array[:, 1].tolist()
    cand_idx = [i for i, s in enumerate(source_ids) if int(s) == source_id]
    if not cand_idx:
        raise SystemExit(f"No test images for source_id={source_id}")
    cand_idx = cand_idx[:n_candidates]
    best_idx, best_n = cand_idx[0], -1
    for idx in cand_idx:
        _, _, targets = test_subset[idx]
        n = len(np.asarray(targets["y"]).reshape(-1, 2))
        if n > best_n:
            best_idx, best_n = idx, n
    print(f"  picked subset index {best_idx} (GT={best_n} points) "
          f"from {len(cand_idx)} candidates")
    return best_idx


def render_panel(source_name, rows, radii, eval_score_threshold,
                 score_thresh, out_path):
    """rows: list of dicts with image, gt, density, and per-radius peaks."""
    n_cols = 2 + len(radii)
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 4.2 * n_rows),
                             squeeze=False)

    for r, row in enumerate(rows):
        sz = row["image_size"]
        img = np.clip(np.transpose(row["image"], (1, 2, 0)), 0, 1)
        H, W = img.shape[:2]
        gt = row["gt"]
        dm = row["density"]
        dm_up = np.array(torch.nn.functional.interpolate(
            torch.from_numpy(dm)[None, None].float(),
            size=(H, W), mode="bilinear", align_corners=False)[0, 0])

        # Col 0: RGB + GT
        ax = axes[r][0]
        ax.imshow(img)
        if len(gt):
            ax.scatter(gt[:, 0], gt[:, 1], s=14, c="cyan",
                       edgecolors="black", linewidths=0.4)
        ax.set_ylabel(f"image_size={sz}\ndensity {dm.shape[0]}x{dm.shape[1]}",
                      fontsize=11)
        if r == 0:
            ax.set_title(f"Image + GT ({len(gt)})", fontsize=11)
        else:
            ax.set_title(f"GT={len(gt)}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # Col 1: raw density map
        ax = axes[r][1]
        im = ax.imshow(dm_up, cmap="magma")
        if r == 0:
            ax.set_title("Raw density map", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])

        # Cols 2..: peaks per radius
        for c, radius in enumerate(radii):
            ax = axes[r][2 + c]
            pk = row["peaks"][radius]
            pxy, psc = pk["xy"], pk["scores"]
            keep = psc > eval_score_threshold
            ax.imshow(img)
            ax.imshow(dm_up, cmap="magma", alpha=0.45)
            if keep.any():
                ax.scatter(pxy[keep, 0], pxy[keep, 1], s=22, marker="o",
                           facecolors="none", edgecolors="lime", linewidths=1.1)
            if (~keep).any():
                ax.scatter(pxy[~keep, 0], pxy[~keep, 1], s=22, marker="x",
                           c="red", linewidths=1.0)
            title = (f"radius={radius}\nkept {int(keep.sum())} / peaks {len(pxy)}"
                     if r == 0 else
                     f"kept {int(keep.sum())} / {len(pxy)}")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"TreeFormer density & points vs image_size and integration radius\n"
        f"{source_name}  (score_thresh={score_thresh}, "
        f"eval_score_threshold={eval_score_threshold:g}; lime=kept, red x=dropped)",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default="weecology/deepforest-tree-point")
    p.add_argument("--revision", type=str, default="main")
    p.add_argument("--root-dir", type=str,
                   default=os.environ.get("MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    p.add_argument("--split-scheme", type=str, default="within-distribution",
                   choices=["within-distribution", "out-of-distribution", "crossgeometry"])
    p.add_argument("--source", action="append", required=True,
                   help="Source name or unique substring. Repeat for multiple.")
    p.add_argument("--image-sizes", type=int, nargs="+", default=[448, 672, 896, 1280])
    p.add_argument("--radii", type=int, nargs="+", default=[1, 2])
    p.add_argument("--score-thresh", type=float, default=0.1)
    p.add_argument("--n-candidates", type=int, default=20,
                   help="Scan this many test images per source; pick the densest.")
    p.add_argument("--image-index", type=int, default=None,
                   help="Override: use this subset index for ALL sources instead "
                        "of auto-picking the densest image.")
    p.add_argument("--output-dir", type=str, default="outputs/density_panels")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.checkpoint, args.revision, args.score_thresh)

    # Rebuild the dataset once per image_size (resize is baked into the dataset).
    datasets = {}
    for sz in args.image_sizes:
        print(f"Loading TreePoints @ image_size={sz} ...")
        datasets[sz] = get_dataset("TreePoints", root_dir=args.root_dir,
                                   split_scheme=args.split_scheme, image_size=sz)
    base_ds = datasets[args.image_sizes[0]]
    eval_score_threshold = base_ds.eval_score_threshold

    for source in args.source:
        source_id, source_name = find_source_id(base_ds, source)
        print(f"\n=== {source_name} (id={source_id}) ===")
        if args.image_index is not None:
            target_idx = args.image_index
        else:
            target_idx = pick_densest_index(base_ds, source_id, args.n_candidates)

        rows = []
        for sz in args.image_sizes:
            test_subset = datasets[sz].get_subset("test")
            _, x, targets = test_subset[target_idx]
            gt = np.asarray(targets["y"]).reshape(-1, 2)
            peaks = {}
            density = None
            for radius in args.radii:
                model.model.score_integration_radius = radius
                pred, dm = predict_with_density(model, x)
                density = dm  # identical across radii
                peaks[radius] = {
                    "xy": pred["points"].detach().cpu().numpy().reshape(-1, 2),
                    "scores": pred["scores"].detach().cpu().numpy().reshape(-1),
                }
            rows.append({"image_size": sz,
                         "image": x.cpu().numpy(),
                         "gt": gt, "density": density, "peaks": peaks})
            kept = {r: int((peaks[r]["scores"] > eval_score_threshold).sum())
                    for r in args.radii}
            print(f"  size={sz:5d} GT={len(gt):4d} "
                  f"peaks={ {r: len(peaks[r]['xy']) for r in args.radii} } "
                  f"kept={kept}")

        safe = "".join(c if c.isalnum() else "_" for c in source_name)
        out_path = os.path.join(args.output_dir, f"density_panel_{safe}.png")
        render_panel(source_name, rows, args.radii, eval_score_threshold,
                     args.score_thresh, out_path)


if __name__ == "__main__":
    main()
