"""
9-panel figure: 3 annotation types (rows) × 3 sources (columns).
RGB images with annotation overlays; no masks. Source label in each panel.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image
from shapely import wkt

BASE = '/orange/ewhite/web/public/MillionTrees'

ANNOT_COLOR = '#9b5de5'   # purple – matches completeness figure
LABEL_COLOR = '#1a1a2e'   # dark navy
BOX_LW      = 1.0
POLY_LW     = 1.0
POINT_R     = 4           # point radius in pixels (drawn as circle patches)

# ── panel definitions ────────────────────────────────────────────────────────
# (dataset, source_in_csv, display_label, filename)
PANELS = [
    # row 0: Tree Boxes
    ('TreeBoxes', 'SelvaBox',
     'H Baudchon et al. 2025',
     '20231208_asforestnorthe2_m3m_rgb_tile_valid_1777_gr0p045_7992_1776.png'),
    ('TreeBoxes', 'Weecology_University_Florida',
     'Weinstein et al. 2020',
     'Pfinztal_10.png'),
    ('TreeBoxes', 'Radogoshi et al. 2021',
     'Radogoshi et al. 2021',
     'B09_0098.JPG'),

    # row 1: Tree Points
    ('TreePoints', 'Amirkolaee et al. 2023',
     'Amirkolaee et al. 2023',
     'IMG_671.jpg'),
    ('TreePoints', 'Ventura et al. 2022',
     'Ventura et al. 2022',
     'eureka_2020_20.tif'),
    ('TreePoints', 'Beery et al. 2022',
     'Beery et al. 2022',
     'calgary_cell_4906_imagery.tif'),

    # row 2: Tree Polygons
    ('TreePolygons', 'Firoze et al. 2023',
     'Firoze et al. 2023',
     '33_1.png'),
    ('TreePolygons', 'Troles et al. 2024',
     'Troles et al. 2024',
     'Stadtwald_31_389.tif'),
    ('TreePolygons', 'Jansen et al. 2023',
     'Jansen et al. 2023',
     'PlotS2_20210621_RGB_62.png'),
]

ROW_LABELS = ['Tree Boxes', 'Tree Points', 'Tree Polygons']


def draw_boxes(ax, df_img):
    for _, row in df_img.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax']).iterrows():
        x1, y1 = int(row['xmin']), int(row['ymin'])
        bw = int(row['xmax']) - x1
        bh = int(row['ymax']) - y1
        ax.add_patch(mpatches.Rectangle(
            (x1, y1), bw, bh,
            linewidth=BOX_LW, edgecolor=ANNOT_COLOR, facecolor='none',
        ))


def draw_points(ax, df_img, img_w):
    r = max(3, img_w // 150)
    for _, row in df_img.dropna(subset=['x', 'y']).iterrows():
        ax.add_patch(mpatches.Circle(
            (float(row['x']), float(row['y'])), radius=r,
            color=ANNOT_COLOR, linewidth=0,
        ))


def draw_polygons(ax, df_img):
    for _, row in df_img.dropna(subset=['polygon']).iterrows():
        try:
            geom = wkt.loads(row['polygon'])
            coords = np.array(geom.exterior.coords)
            ax.add_patch(MplPolygon(
                coords, closed=True,
                linewidth=POLY_LW, edgecolor=ANNOT_COLOR, facecolor='none',
            ))
        except Exception:
            continue


def make_panel(ax, ds, source_csv, label, fname):
    img_path = os.path.join(BASE, f'{ds}_v0.12', 'images', fname)
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    ax.imshow(np.array(img))
    ax.set_axis_off()

    df = pd.read_csv(os.path.join(BASE, f'{ds}_v0.12', 'within-distribution.csv'), low_memory=False)
    df_img = df[df['filename'] == fname]

    if ds == 'TreeBoxes':
        draw_boxes(ax, df_img)
    elif ds == 'TreePoints':
        draw_points(ax, df_img, w)
    elif ds == 'TreePolygons':
        draw_polygons(ax, df_img)

    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes, ha='left', va='top',
        fontsize=8.5, fontweight='bold', color='white',
        bbox=dict(facecolor='black', alpha=0.55, pad=2, linewidth=0),
    )


def main():
    fig, axes = plt.subplots(3, 3, figsize=(14, 14), facecolor='white')
    fig.subplots_adjust(wspace=0.04, hspace=0.06)

    # cache CSVs to avoid re-reading
    csv_cache = {}
    for ds in ['TreeBoxes', 'TreePoints', 'TreePolygons']:
        csv_cache[ds] = pd.read_csv(
            os.path.join(BASE, f'{ds}_v0.12', 'within-distribution.csv'), low_memory=False)

    for idx, (ds, source_csv, label, fname) in enumerate(PANELS):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        img_path = os.path.join(BASE, f'{ds}_v0.12', 'images', fname)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        ax.imshow(np.array(img))
        ax.set_axis_off()

        df_img = csv_cache[ds][csv_cache[ds]['filename'] == fname]

        if ds == 'TreeBoxes':
            draw_boxes(ax, df_img)
        elif ds == 'TreePoints':
            draw_points(ax, df_img, w)
        elif ds == 'TreePolygons':
            draw_polygons(ax, df_img)

        ax.text(
            0.02, 0.98, label,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, fontweight='bold', color='white',
            bbox=dict(facecolor='black', alpha=0.55, pad=2, linewidth=0),
        )

    # Row labels
    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].set_ylabel(
            label, fontsize=11, fontweight='bold',
            color=LABEL_COLOR, labelpad=8,
        )
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].tick_params(left=False, labelleft=False)
        axes[row, 0].spines[:].set_visible(False)

    out_base = '/blue/ewhite/b.weinstein/src/MillionTrees/annotation_examples_figure'
    plt.savefig(out_base + '.svg', format='svg', bbox_inches='tight', dpi=150)
    plt.savefig(out_base + '.png', format='png', bbox_inches='tight', dpi=150)
    print(f'Saved {out_base}.svg / .png')


if __name__ == '__main__':
    main()
