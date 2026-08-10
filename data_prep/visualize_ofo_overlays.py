#!/usr/bin/env python3
"""Generate QC overlay plots for OFO unsupervised annotations after coord fix.

Shows both field and unsupervised annotations on the same tile images
so reviewers can compare density, placement, and CHM-detection coverage.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import pandas as pd

FIELD_CSV   = '/orange/ewhite/DeepForest/OpenForestObservatory/field/TreePoints_OFO_field.csv'
UNSUP_CSV   = '/orange/ewhite/DeepForest/OpenForestObservatory/images/TreePoints_OFO_unsupervised.csv'
FIELD_IMGS  = '/orange/ewhite/DeepForest/OpenForestObservatory/field/images'
UNSUP_IMGS  = '/orange/ewhite/DeepForest/OpenForestObservatory/images'
SAVE_DIR    = '/blue/ewhite/b.weinstein/src/MillionTrees/docs/public/ofo_overlays'


def load_data():
    field = pd.read_csv(FIELD_CSV)
    unsup = pd.read_csv(UNSUP_CSV)
    unsup['mission_id_norm'] = unsup['image_path'].apply(
        lambda p: int(os.path.basename(p).split('_')[0])
    )
    return field, unsup


def find_shared_tiles(field, unsup, missions=None, min_both=3):
    """Return list of (mission_id_int, tile_basename, n_field, n_unsup)."""
    if missions is None:
        missions = sorted(
            set(field['mission_id'].astype(int)) & set(unsup['mission_id_norm'])
        )
    shared = []
    for m in missions:
        f_sub = field[field['mission_id'] == m]
        u_sub = unsup[unsup['mission_id_norm'] == m]
        f_tiles = set(f_sub['image_path'].apply(os.path.basename))
        u_tiles = set(u_sub['image_path'].apply(os.path.basename))
        for tile in f_tiles & u_tiles:
            nf = (f_sub['image_path'].apply(os.path.basename) == tile).sum()
            nu = (u_sub['image_path'].apply(os.path.basename) == tile).sum()
            if nf >= min_both and nu >= min_both:
                shared.append((m, tile, int(nf), int(nu)))
    return sorted(shared, key=lambda r: r[2] + r[3], reverse=True)


def plot_overlay_grid(field, unsup, tiles, savepath, n_cols=3):
    """Plot a grid of tiles showing field (cyan) and unsupervised (red) points."""
    n = len(tiles)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    for idx, (mission, tile, nf, nu) in enumerate(tiles):
        ax = axes[idx // n_cols][idx % n_cols]
        img_path = os.path.join(FIELD_IMGS, tile)
        if not os.path.exists(img_path):
            img_path = os.path.join(UNSUP_IMGS, tile)
        img = np.array(Image.open(img_path))
        ax.imshow(img)

        f_ann = field[field['image_path'].apply(os.path.basename) == tile]
        u_ann = unsup[unsup['image_path'].apply(os.path.basename) == tile]

        if len(u_ann):
            ax.scatter(u_ann['x'], u_ann['y'], c='red', s=30,
                       edgecolors='darkred', lw=0.5,
                       label=f'CHM unsup ({len(u_ann)})', zorder=3, alpha=0.75)
        if len(f_ann):
            ax.scatter(f_ann['x'], f_ann['y'], c='cyan', s=35,
                       edgecolors='blue', lw=0.5,
                       label=f'Field ({len(f_ann)})', zorder=4, alpha=0.9)

        ax.set_title(f'M{mission}  {tile}\nField={nf}  CHM={nu}', fontsize=8)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.6)
        ax.axis('off')

    # Hide unused axes
    for idx in range(len(tiles), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    fig.suptitle(
        'OFO Field (cyan) vs CHM-derived Unsupervised (red)\n'
        'Shared tiles on shared missions — tile-local coordinates after fix',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


def plot_coord_check(unsup, savepath):
    """Quick sanity plot: histogram of x,y ranges to confirm they're all in [0,800]."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(unsup['x'], bins=80, color='steelblue', alpha=0.7)
    axes[0].axvline(0, color='red', lw=1); axes[0].axvline(800, color='red', lw=1)
    axes[0].set_title('x coordinates (should be in [0, 800])'); axes[0].set_xlabel('x (px)')
    axes[1].hist(unsup['y'], bins=80, color='tomato', alpha=0.7)
    axes[1].axvline(0, color='red', lw=1); axes[1].axvline(800, color='red', lw=1)
    axes[1].set_title('y coordinates (should be in [0, 800])'); axes[1].set_xlabel('y (px)')
    fig.suptitle('Unsupervised CSV — coordinate distribution after fix', fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {savepath}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    field, unsup = load_data()

    # Sanity: confirm x,y exist and are in range
    assert 'x' in unsup.columns and 'y' in unsup.columns, "x/y columns missing!"
    pct_ok = ((unsup['x'].between(0, 800)) & (unsup['y'].between(0, 800))).mean()
    print(f"Annotations with x,y in [0,800]: {pct_ok*100:.2f}%")
    print(f"Total annotations: {len(unsup):,}")

    plot_coord_check(unsup, os.path.join(SAVE_DIR, 'ofo_unsup_coord_check.png'))

    # Sample overlays: pick best tiles from varied missions
    target_missions = [89, 96, 109, 124, 128, 131, 147, 153, 161, 168, 180, 184]
    shared = find_shared_tiles(field, unsup, missions=target_missions, min_both=5)
    print(f"Found {len(shared)} shared tiles with ≥5 annotations each side")

    # Best tile per mission (most combined annotations)
    seen = {}
    best_tiles = []
    for entry in shared:
        m = entry[0]
        if m not in seen:
            seen[m] = entry
            best_tiles.append(entry)

    best_tiles = best_tiles[:9]  # 3×3 grid
    print("Plotting tiles:", [(m, t) for m, t, *_ in best_tiles])

    plot_overlay_grid(field, unsup, best_tiles,
                      os.path.join(SAVE_DIR, 'ofo_fixed_overlays.png'), n_cols=3)


if __name__ == '__main__':
    main()
