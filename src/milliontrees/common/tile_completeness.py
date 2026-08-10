"""Per-tile annotation completeness: which eval tiles are annotated wall to wall.

The packaged ``complete`` flag is per *source* (``data_prep/source_completeness.csv``), which
is too coarse for the held-out validation sources. Their annotations come from a TLS plot
footprint that does not line up with the tiling grid, so a source is a mix of interior tiles
annotated edge to edge and **edge tiles** where the footprint clips a corner and the rest of
the tile is unlabelled forest. Scoring AP against an edge tile measures the annotation
protocol, not the model: every correct detection outside the footprint is a false positive.
See ``docs/validation_ap_completeness.md``.

This module defines that distinction once, as a deterministic function of data already in the
package (annotation geometry + the precomputed tree-coverage masks), so the audit script, the
loaders and the packaging step all draw the same line. Nothing is hand-curated.

A tile is complete when both hold:

* ``max_margin <= max_margin_tol`` -- no side of the tile has a strip wider than this fraction
  sitting outside the annotated footprint, i.e. the annotations reach every edge.
* ``canopy_annotated_frac >= min_canopy_annotated`` -- at least this share of the pixels an
  independent canopy segmenter (Restor TCD, shipped in ``masks/``) calls tree fall inside an
  annotation. This catches the case a margin test misses: annotations that span the tile but
  leave an unlabelled block of forest in the middle.

Both tests need annotations with *extent*, so the flag is derived once from the box or polygon
package and shared with the point package for the same tile -- a stem point sits half a crown
inside the tile edge and covers no canopy pixels, so scoring points directly would reject tiles
the box version accepts. ``data_prep/flag_tile_completeness.py`` generates the shared manifest;
``MANIFEST_PATH`` is where the loaders look for it.
"""

from pathlib import Path

import numpy as np

# Defaults chosen from the v0.22 validation audit (docs/validation_completeness_*.csv):
# the 35 interior Frey tiles sit at margin 0.00 / canopy 0.99, the 26 edge tiles at margin
# 0.05-0.94 / canopy 0.01-0.89, and the 24 Allen tiles at margin <=0.024. The canopy floor is
# deliberately loose -- it is there to reject tiles that are mostly unlabelled forest, not to
# grade annotation quality, and open-canopy sites (Allen's Spanish plots) legitimately score
# 0.56-0.89 because the segmenter counts shrub the annotators did not.
DEFAULT_MAX_MARGIN = 0.025
DEFAULT_MIN_CANOPY_ANNOTATED = 0.5


def geometry_bounds(df):
    """Per-row ``(xmin, ymin, xmax, ymax)`` for boxes, polygon WKT, or points."""
    if {"xmin", "ymin", "xmax", "ymax"}.issubset(df.columns):
        return df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    if "polygon" in df.columns:
        from shapely import wkt
        return np.array([wkt.loads(p).bounds for p in df["polygon"]],
                        dtype=float)
    if {"x", "y"}.issubset(df.columns):
        xy = df[["x", "y"]].to_numpy(dtype=float)
        return np.concatenate([xy, xy], axis=1)
    raise ValueError(
        f"No usable geometry columns in {list(df.columns)}; expected box, polygon or point."
    )


def annotated_footprint_mask(bounds, shape):
    """Boolean ``[H, W]`` union of the annotation bounding boxes."""
    height, width = int(shape[0]), int(shape[1])
    covered = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in bounds:
        xa, xb = int(max(0, np.floor(x0))), int(min(width, np.ceil(x1)))
        ya, yb = int(max(0, np.floor(y0))), int(min(height, np.ceil(y1)))
        if xb > xa and yb > ya:
            covered[ya:yb, xa:xb] = True
    return covered


def max_margin(bounds, width, height):
    """Widest unannotated strip on any side, as a fraction of the tile."""
    if len(bounds) == 0 or not width or not height:
        return float("nan")
    x0, y0 = bounds[:, 0].min(), bounds[:, 1].min()
    x1, y1 = bounds[:, 2].max(), bounds[:, 3].max()
    return float(
        max(x0 / width, y0 / height, (width - x1) / width,
            (height - y1) / height))


def has_extent(bounds):
    """Whether the geometry covers area.

    Points are stored as degenerate boxes.
    """
    return len(bounds) > 0 and bool(
        np.any((bounds[:, 2] > bounds[:, 0]) & (bounds[:, 3] > bounds[:, 1])))


def canopy_annotated_fraction(bounds, tree_mask):
    """Share of canopy pixels that fall inside at least one annotation."""
    if tree_mask is None or not has_extent(bounds):
        return float("nan")
    total = int(np.asarray(tree_mask, dtype=bool).sum())
    if total == 0:
        return float("nan")
    covered = annotated_footprint_mask(bounds, tree_mask.shape)
    return float((covered & np.asarray(tree_mask, dtype=bool)).sum() / total)


def tile_stats(bounds, width, height, tree_mask=None):
    """Completeness statistics for one tile."""
    return {
        "max_margin": max_margin(bounds, width, height),
        "canopy_annotated_frac": canopy_annotated_fraction(bounds, tree_mask),
    }


def is_complete_tile(stats,
                     max_margin_tol=DEFAULT_MAX_MARGIN,
                     min_canopy_annotated=DEFAULT_MIN_CANOPY_ANNOTATED):
    """Apply the completeness rule to one tile's statistics.

    A NaN canopy fraction (no mask, or a mask with no canopy) means the canopy test cannot be
    evaluated, so the margin test decides on its own rather than failing the tile.
    """
    margin = stats.get("max_margin", float("nan"))
    if not np.isfinite(margin) or margin > max_margin_tol:
        return False
    canopy = stats.get("canopy_annotated_frac", float("nan"))
    if np.isfinite(canopy) and canopy < min_canopy_annotated:
        return False
    return True


def _read_size(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return im.size  # (width, height)


def _read_mask(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    if not Path(path).exists():
        return None
    with Image.open(path) as mk:
        return np.asarray(mk.convert("L")) > 0


def score_tiles(df,
                data_dir=None,
                use_masks=True,
                progress=False,
                image_path_col=None,
                mask_dir=None,
                strict_masks=False):
    """Completeness statistics for every image in ``df``, one row per image.

    ``df`` is a frame of annotations with a ``filename`` column. Images and masks are located
    either way:

    * ``data_dir`` -- a packaged release, i.e. ``<data_dir>/images`` and ``<data_dir>/masks``.
      This is what the loaders and the audit script use.
    * ``image_path_col`` + ``mask_dir`` -- explicit source paths, for the packaging step, which
      has to decide what to ship *before* any image is copied into the release.

    Returns ``filename, source, n_annotations, width, height, max_margin,
    canopy_annotated_frac``.
    """
    import pandas as pd

    if data_dir is not None:
        image_dir = Path(data_dir) / "images"
        mask_dir = Path(data_dir) / "masks"
    elif image_path_col is None or mask_dir is None:
        raise ValueError(
            "Pass either data_dir, or both image_path_col and mask_dir.")
    else:
        image_dir = None
        mask_dir = Path(mask_dir)

    rows = []
    for i, (fname, group) in enumerate(df.groupby("filename", sort=True)):
        image_path = (image_dir / fname if image_dir is not None else Path(
            group[image_path_col].iloc[0]))
        width, height = _read_size(image_path)
        bounds = geometry_bounds(group)
        tree_mask = None
        if use_masks:
            tree_mask = _read_mask(mask_dir / f"{Path(fname).stem}.png")
            if tree_mask is not None and tree_mask.shape != (height, width):
                if strict_masks:
                    # Packaging path: shipping this pair would make the release unloadable,
                    # since get_tree_coverage_mask raises on a size mismatch. The mask
                    # precompute skips images that already have a mask, so a re-tiled source
                    # silently keeps the previous tiling's mask -- exactly how v0.21 broke.
                    raise ValueError(
                        f"Stale tree-coverage mask for {fname}: mask is "
                        f"{tree_mask.shape[1]}x{tree_mask.shape[0]} but the image is "
                        f"{width}x{height}. Regenerate masks for this source "
                        f"(precompute_tree_coverage_masks --overwrite) before packaging."
                    )
                # Analysis path: fall back to the margin test rather than silently scoring
                # against the wrong raster.
                tree_mask = None
        rows.append(
            dict(filename=fname,
                 source=group["source"].iloc[0],
                 n_annotations=len(group),
                 width=width,
                 height=height,
                 **tile_stats(bounds, width, height, tree_mask)))
        if progress and (i + 1) % 50 == 0:
            print(f"  scored {i + 1} tiles")
    return pd.DataFrame(rows)


def complete_tile_names(df,
                        data_dir,
                        max_margin_tol=DEFAULT_MAX_MARGIN,
                        min_canopy_annotated=DEFAULT_MIN_CANOPY_ANNOTATED,
                        use_masks=True):
    """Set of filenames in ``df`` that pass the completeness rule."""
    stats = score_tiles(df, data_dir, use_masks=use_masks)
    keep = stats.apply(
        lambda r: is_complete_tile(r, max_margin_tol, min_canopy_annotated),
        axis=1)
    return set(stats.loc[keep, "filename"])


# --------------------------------------------------------------------------- #
# Shared manifest
# --------------------------------------------------------------------------- #
# Generated by data_prep/flag_tile_completeness.py, version-controlled so the exact tile set
# behind a published number is reviewable, and read by all three loaders so the point task
# evaluates the same tiles as the box and polygon tasks.
MANIFEST_PATH = (Path(__file__).resolve().parents[3] / "data_prep" /
                 "tile_completeness.csv")

MANIFEST_COLUMNS = [
    "version", "split", "filename", "source", "n_annotations", "max_margin",
    "canopy_annotated_frac", "complete"
]


def load_manifest(path=None):
    """Read the tile-completeness manifest, or None if it is not present."""
    import pandas as pd

    path = Path(path or MANIFEST_PATH)
    if not path.exists():
        return None
    manifest = pd.read_csv(path, dtype={"version": str})
    manifest["complete"] = (
        manifest["complete"].astype(str).str.strip().str.lower() == "true")
    return manifest


def manifest_complete_tiles(version, splits, path=None):
    """Complete filenames for ``version``/``splits`` from the manifest.

    Returns ``None`` when the manifest has no entry for that version and split, so callers can fall
    back to computing the rule directly instead of silently keeping every tile.
    """
    manifest = load_manifest(path)
    if manifest is None:
        return None
    rows = manifest[(manifest["version"] == str(version)) &
                    (manifest["split"].isin(list(splits)))]
    if rows.empty:
        return None
    return set(rows.loc[rows["complete"], "filename"])
