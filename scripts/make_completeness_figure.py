"""
Generate a figure showing annotation completeness examples.
Two columns (High / Low completeness), each with:
  - top:    RGB image with bounding-box annotations overlaid
  - bottom: 3-class mask  (annotated trees | unannotated trees | background)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from PIL import Image

DATA_DIR = '/orange/ewhite/web/public/MillionTrees/TreeBoxes_v0.12'
IMG_DIR  = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
CSV_PATH = os.path.join(DATA_DIR, 'random.csv')
OUT_SVG  = '/blue/ewhite/b.weinstein/src/MillionTrees/annotation_completeness_figure.svg'
OUT_PNG  = '/blue/ewhite/b.weinstein/src/MillionTrees/annotation_completeness_figure.png'

# ── selected examples ────────────────────────────────────────────────────────
HIGH_FILE   = 'B10_0073.JPG'
HIGH_SOURCE = 'Radogoshi et al. 2021'

LOW_FILE    = 'Flora Pluas RGB_16_11400_3800_15400_7800.png'
LOW_SOURCE  = 'Reiersen et al. 2022'

# ── colours ──────────────────────────────────────────────────────────────────
BOX_COLOR        = '#9b5de5'           # purple – box outlines on RGB
BOX_LW           = 1.5

C_BACKGROUND     = np.array([26,  26,  46],  dtype=np.uint8)   # dark navy
C_UNANNOTATED    = np.array([142, 202, 230],  dtype=np.uint8)   # muted blue  – tree, no box
C_ANNOTATED      = np.array([82,  183, 136],  dtype=np.uint8)   # green       – tree + box

LABEL_COLOR = '#1a1a2e'


def build_three_class_mask(df_img, w, h, tree_mask):
    """Return an RGB array with three classes."""
    box_mask = np.zeros((h, w), dtype=bool)
    for _, row in df_img.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax']).iterrows():
        x1, y1 = max(0, int(row['xmin'])), max(0, int(row['ymin']))
        x2, y2 = min(w, int(row['xmax'])), min(h, int(row['ymax']))
        if x2 > x1 and y2 > y1:
            box_mask[y1:y2, x1:x2] = True

    annotated   = tree_mask & box_mask
    unannotated = tree_mask & ~box_mask

    rgb = np.full((h, w, 3), C_BACKGROUND, dtype=np.uint8)
    rgb[unannotated] = C_UNANNOTATED
    rgb[annotated]   = C_ANNOTATED

    pct = annotated.sum() / tree_mask.sum() if tree_mask.sum() > 0 else 0.0
    return rgb, pct


def draw_rgb_panel(ax, img_arr, df_img, source, title):
    ax.imshow(img_arr)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight='bold', color=LABEL_COLOR, pad=5)

    h, w = img_arr.shape[:2]
    for _, row in df_img.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax']).iterrows():
        x1, y1 = int(row['xmin']), int(row['ymin'])
        bw = int(row['xmax']) - x1
        bh = int(row['ymax']) - y1
        rect = mpatches.Rectangle(
            (x1, y1), bw, bh,
            linewidth=BOX_LW, edgecolor=BOX_COLOR, facecolor='none',
        )
        ax.add_patch(rect)

    ax.text(
        0.02, 0.98, source,
        transform=ax.transAxes, ha='left', va='top',
        fontsize=8.5, fontweight='bold', color='white',
        bbox=dict(facecolor='black', alpha=0.55, pad=2, linewidth=0),
    )


def draw_mask_panel(ax, mask_rgb, df_img, pct_annotated):
    ax.imshow(mask_rgb)
    ax.set_axis_off()

    for _, row in df_img.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax']).iterrows():
        x1, y1 = int(row['xmin']), int(row['ymin'])
        bw = int(row['xmax']) - x1
        bh = int(row['ymax']) - y1
        rect = mpatches.Rectangle(
            (x1, y1), bw, bh,
            linewidth=BOX_LW, edgecolor=BOX_COLOR, facecolor='none',
        )
        ax.add_patch(rect)

    ax.text(
        0.02, 0.02, f'{pct_annotated:.0%} of tree pixels annotated',
        transform=ax.transAxes, ha='left', va='bottom',
        fontsize=8.5, color='white',
        bbox=dict(facecolor='black', alpha=0.55, pad=2, linewidth=0),
    )


def main():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    panels = [
        (HIGH_FILE, HIGH_SOURCE, 'High completeness'),
        (LOW_FILE,  LOW_SOURCE,  'Low completeness'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='white')
    fig.subplots_adjust(wspace=0.04, hspace=0.06)

    for col, (fname, source, title) in enumerate(panels):
        img = Image.open(os.path.join(IMG_DIR, fname)).convert('RGB')
        w, h = img.size
        img_arr = np.array(img)

        mask_fname = os.path.splitext(fname)[0] + '.png'
        tree_mask_path = os.path.join(MASK_DIR, mask_fname)
        tree_mask = np.array(Image.open(tree_mask_path).convert('L')) > 0

        df_img = df[df['filename'] == fname]
        mask_rgb, pct = build_three_class_mask(df_img, w, h, tree_mask)

        draw_rgb_panel(axes[0, col], img_arr, df_img, source, title)
        draw_mask_panel(axes[1, col], mask_rgb, df_img, pct)

    # Row labels
    for ax, label in zip(axes[:, 0], ['RGB Image', 'Annotation Mask']):
        ax.set_ylabel(label, fontsize=10, labelpad=6, color=LABEL_COLOR)
        ax.yaxis.set_visible(True)
        ax.tick_params(left=False, labelleft=False)
        ax.spines[:].set_visible(False)

    # Legend for the mask
    legend_elements = [
        Patch(facecolor=C_ANNOTATED/255,   label='Annotated pixel'),
        Patch(facecolor=C_UNANNOTATED/255, label='Unannotated pixel'),
        Patch(facecolor=C_BACKGROUND/255,  label='Background'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        fontsize=9, frameon=True,
        bbox_to_anchor=(0.5, 0.01), ncol=3,
    )

    plt.savefig(OUT_SVG, format='svg', bbox_inches='tight', dpi=150)
    plt.savefig(OUT_PNG, format='png', bbox_inches='tight', dpi=150)
    print(f'Saved {OUT_SVG}')
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
