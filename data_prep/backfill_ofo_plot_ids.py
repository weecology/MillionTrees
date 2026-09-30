#!/usr/bin/env python3
"""Backfill ``plot_id`` and ``license`` onto an already-tiled OFO field annotation CSV.

``process_ofo_field.py`` now carries the ground-reference ``plot_id`` through tiling and
stamps each annotation with its license (see ``ofo_catalog_licenses``), but re-running it
means re-downloading and re-tiling every mission orthomosaic -- a ~12 h bigmem job that
rewrites images the current release already ships.

The tiling is reproducible from georeferencing alone, so the same columns can be recovered
offline instead. For each mission this rebuilds the ``split_raster`` windows (patch 800, no
overlap), projects every field stem into ortho pixels, translates it into the window it
lands in exactly as ``select_annotations`` does, and joins back to the packaged rows on
``(filename, x, y)``.

The join must be total: the script fails rather than write a partly-licensed CSV, since a
row with no license would silently fall back to the source-level CC-BY-NC-SA.

Usage::

    python data_prep/backfill_ofo_plot_ids.py \\
        --annotations /orange/ewhite/DeepForest/OpenForestObservatory/field/TreePoints_OFO_field.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import rasterio as rio
import slidingwindow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ofo_catalog_licenses import load_combined_licenses  # noqa: E402
from process_ofo_field import load_field_trees  # noqa: E402

FIELD_TREES = ("/orange/ewhite/DeepForest/OpenForestObservatory/field/"
               "OFO-field-trees-registered-to-drone.gpkg")
OFO_ROOT = "/orange/ewhite/DeepForest/OpenForestObservatory/missions_03"
PATCH_SIZE = 800

#: The rebuilt coordinates and the packaged ones come from the same arithmetic, so a match is
#: exact to floating-point noise. Joining on rounded keys instead would drop the handful of
#: stems that sit on a rounding boundary, so pair them by distance and demand near-zero.
TOLERANCE_PX = 1e-3


def rebuild_stem_table(field_trees_path=FIELD_TREES, ofo_root=OFO_ROOT):
    """One row per (tile, stem): filename, tile-local x/y, and the stem's plot_id."""
    trees = load_field_trees(field_trees_path)
    trees["plot_id"] = trees["plot_id"].astype(str).str.zfill(4)

    records = []
    for mission_id, mission_trees in trees.groupby("mission_id"):
        ortho = os.path.join(ofo_root, mission_id, "photogrammetry_03", "full",
                             f"{mission_id}_ortho.tif")
        if not os.path.exists(ortho):
            raise FileNotFoundError(
                f"{ortho} is missing, so mission {mission_id} cannot be reproduced."
            )
        with rio.open(ortho) as src:
            width, height, transform, crs = (src.width, src.height,
                                             src.transform, src.crs)
        if height < PATCH_SIZE or width < PATCH_SIZE:
            continue  # process_ofo_field skips these missions entirely

        if (mission_trees.crs is not None and crs is not None and
                mission_trees.crs != crs):
            mission_trees = mission_trees.to_crs(crs)

        col, row = ~transform * (mission_trees.geometry.x.values,
                                 mission_trees.geometry.y.values)
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        col, row = col[inside], row[inside]
        plot_ids = mission_trees["plot_id"].values[inside]

        windows = slidingwindow.generateForSize(
            width, height, slidingwindow.DimOrder.ChannelHeightWidth,
            PATCH_SIZE, 0)
        for index, window in enumerate(windows):
            x0, y0, w, h = window.getRect()
            hit = (col >= x0) & (col <= x0 + w) & (row >= y0) & (row <= y0 + h)
            if not hit.any():
                continue  # split_raster(allow_empty=False) writes no tile here
            records.append(
                pd.DataFrame({
                    "filename": f"{mission_id}_ortho_{index}.png",
                    # select_annotations translates points to the window origin.
                    "x": col[hit] - x0,
                    "y": row[hit] - y0,
                    "plot_id": plot_ids[hit],
                }))
    return pd.concat(records, ignore_index=True)


def backfill(annotations_path, output_path=None, dry_run=False):
    annotations = pd.read_csv(annotations_path)
    annotations["filename"] = annotations["image_path"].map(os.path.basename)

    stems = rebuild_stem_table()
    stems_by_tile = {name: group for name, group in stems.groupby("filename")}

    plot_id = pd.Series(pd.NA, index=annotations.index, dtype=object)
    worst = 0.0
    ambiguous = []
    for name, rows in annotations.groupby("filename"):
        candidates = stems_by_tile.get(name)
        if candidates is None:
            continue
        cx = candidates["x"].to_numpy()
        cy = candidates["y"].to_numpy()
        cplot = candidates["plot_id"].to_numpy()
        distance = np.hypot(rows["x"].to_numpy()[:, None] - cx[None, :],
                            rows["y"].to_numpy()[:, None] - cy[None, :])
        nearest = distance.argmin(axis=1)
        closest = distance[np.arange(len(rows)), nearest]
        matched = closest <= TOLERANCE_PX
        if matched.any():
            worst = max(worst, float(closest[matched].max()))
        # A stem sitting on top of another from a different plot would make the
        # license ambiguous rather than merely redundant.
        for i in np.flatnonzero(matched):
            tied = cplot[distance[i] <= TOLERANCE_PX]
            if len(set(tied)) > 1:
                ambiguous.append((name, sorted(set(tied))))
        plot_id.loc[rows.index[matched]] = cplot[nearest[matched]]

    if ambiguous:
        raise ValueError(
            f"{len(ambiguous)} annotations sit on stems from two different plots, e.g. "
            f"{ambiguous[:3]}; the license cannot be resolved by position.")

    merged = annotations.copy()
    merged["plot_id"] = plot_id
    unmatched = merged["plot_id"].isna()
    if unmatched.any():
        sample = merged.loc[unmatched, "filename"].head(5).tolist()
        raise ValueError(
            f"{int(unmatched.sum())}/{len(merged)} annotations did not match a rebuilt "
            f"stem, e.g. {sample}. The tiling assumptions no longer hold; rerun "
            "process_ofo_field.py instead of backfilling.")
    print(f"matched every annotation to a stem; worst offset {worst:.2e} px")

    licenses = load_combined_licenses()
    missing = sorted(set(merged["plot_id"]) - set(licenses))
    if missing:
        raise ValueError(f"Plot(s) {missing} are not in the OFO plot catalog.")
    merged["license"] = merged["plot_id"].map(licenses)

    merged = merged.drop(columns=["filename"])
    print(f"{len(merged)} annotations matched across "
          f"{merged['plot_id'].nunique()} plots")
    print(merged["license"].value_counts().to_string())

    if dry_run:
        return merged
    destination = output_path or annotations_path
    merged.to_csv(destination, index=False)
    print(f"wrote {destination}")
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output",
                        help="Defaults to overwriting --annotations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    backfill(args.annotations, args.output, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
