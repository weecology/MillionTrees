#!/usr/bin/env python3
"""Regenerate the per-tile OFO license overrides in ``license_data/overrides.csv``.

Open Forest Observatory is the one MillionTrees source whose imagery and annotations come
from *different* upstream datasets with *different* owners, so a single source-level license
cannot describe it:

* The image tile is cut from an OFO drone mission orthomosaic. Every mission in the OFO
  catalog is CC-BY 4.0.
* The points are a contributor's ground-reference stem map. Those plots span five licenses
  (CC-BY 4.0, CC-BY-SA 4.0, CC-BY-NC-SA 4.0, CC0 1.0, and U.S. Forest Service public domain).

A packaged row ships both, so the license recorded for it is the **most restrictive of the
two**. Combining CC-BY 4.0 imagery with each plot license gives:

    plot CC0 1.0          -> CC-BY-4.0        (the image still requires attribution)
    plot public domain    -> CC-BY-4.0
    plot CC BY 4.0        -> CC-BY-4.0
    plot CC BY SA 4.0     -> CC-BY-SA-4.0
    plot CC BY NC SA 4.0  -> CC-BY-NC-SA-4.0

The plot's own license is kept in the ``notes`` column so the provenance is not lost.

Tile attribution
----------------
The packaged filename is ``{mission_id}_ortho_{window}_OFO_field_2025.png``, and one mission
can cover plots under different licenses, so a per-mission rule is not always enough. This
script rebuilds the ``deepforest.preprocess.split_raster`` tiling (patch_size 800, no overlap)
from each orthomosaic's georeferencing alone -- no pixels are read -- and assigns every field
stem to the window it falls in, which gives the plot(s) behind each tile. A tile spanning two
plots takes the more restrictive of their licenses.

``--validate`` checks the rebuilt tile set against a packaged release; it must reproduce the
release's OFO filenames exactly, or the tiling assumptions no longer hold and the emitted
table would be guesswork.

Usage::

    python data_prep/ofo_license_overrides.py \
        --validate /orange/ewhite/web/public/MillionTrees/TreePoints_v0.25
"""

import argparse
import glob
import os
import sys

import pandas as pd
import rasterio as rio
import slidingwindow

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ofo_catalog_licenses import (
    COMBINED_WITH_IMAGERY,  # noqa: E402
    DEFAULT_CACHE_DIR,
    MISSIONS_GPKG,
    PLOTS_GPKG,
    RESTRICTIVENESS,
    ZENODO_RECORD,
    check_mission_imagery_license,
    download_catalog,
    plot_licenses)
from process_ofo_field import load_field_trees  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERRIDES_PATH = os.path.join(REPO, "src", "milliontrees", "common",
                              "license_data", "overrides.csv")

FIELD_TREES = ("/orange/ewhite/DeepForest/OpenForestObservatory/field/"
               "OFO-field-trees-registered-to-drone.gpkg")
OFO_ROOT = "/orange/ewhite/DeepForest/OpenForestObservatory/missions_03"
SOURCE = "OFO field 2025"
PATCH_SIZE = 800


def tile_plot_table(licenses_by_plot):
    """One row per (tile, plot) pair, rebuilt from the mission orthomosaics' georeferencing."""
    trees = load_field_trees(FIELD_TREES)
    trees["plot_id"] = trees["plot_id"].astype(str).str.zfill(4)

    missing = sorted(set(trees["plot_id"]) - set(licenses_by_plot))
    if missing:
        raise ValueError(
            f"Plot(s) {missing} are registered to a drone mission but absent from the OFO "
            "plot catalog, so their license is unknown.")

    rows = []
    for mission_id, mission_trees in trees.groupby("mission_id"):
        ortho = os.path.join(OFO_ROOT, mission_id, "photogrammetry_03", "full",
                             f"{mission_id}_ortho.tif")
        if not os.path.exists(ortho):
            raise FileNotFoundError(
                f"{ortho} is missing; rerun slurm/process_ofo_field.sbatch first so the "
                "tiling can be reproduced.")
        with rio.open(ortho) as src:
            width, height, transform, crs = (src.width, src.height,
                                             src.transform, src.crs)
        if height < PATCH_SIZE or width < PATCH_SIZE:
            continue  # process_ofo_field skips these missions entirely

        if (mission_trees.crs is not None and crs is not None and
                mission_trees.crs != crs):
            mission_trees = mission_trees.to_crs(crs)

        # Geo -> ortho pixel, the same projection read_file applies before split_raster.
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
            for plot_id in sorted(set(plot_ids[hit])):
                rows.append({
                    "filename":
                        f"{mission_id}_ortho_{index}_{SOURCE.replace(' ', '_')}.png",
                    "mission_id":
                        mission_id,
                    "plot_id":
                        plot_id,
                    "plot_license":
                        licenses_by_plot[plot_id],
                })
    return pd.DataFrame(rows)


def resolve_tiles(tile_plots):
    """Collapse (tile, plot) pairs to one effective license per tile."""
    rank = {name: i for i, name in enumerate(RESTRICTIVENESS)}
    grouped = tile_plots.groupby("filename").agg(
        mission_id=("mission_id", "first"),
        plot_ids=("plot_id", lambda s: ",".join(sorted(set(s)))),
        plot_license=("plot_license", lambda s: min(s, key=rank.__getitem__)),
    ).reset_index()
    grouped["license"] = grouped["plot_license"].map(COMBINED_WITH_IMAGERY)
    return grouped


def build_rules(tiles):
    """Emit the smallest set of override rules that reproduces ``tiles`` exactly.

    One rule per mission covers the license most of its tiles carry; the minority tiles get an
    exact-filename rule after it, which wins because overrides.csv is applied top to bottom.
    """
    rules = []
    for mission_id, sub in tiles.groupby("mission_id"):
        counts = sub["license"].value_counts()
        majority = counts.index[0]
        plot_note = sorted(
            set(sub.loc[sub["license"].eq(majority), "plot_license"]))
        rules.append({
            "source":
                SOURCE,
            "filename_pattern":
                f"{mission_id}_ortho_*_{SOURCE.replace(' ', '_')}.png",
            "license":
                majority,
            "notes": (f"mission {mission_id}, {counts[majority]} tiles; "
                      f"plots {'/'.join(plot_note)} + CC-BY 4.0 imagery"),
        })
        for row in sub[sub["license"].ne(majority)].sort_values(
                "filename").itertuples():
            rules.append({
                "source":
                    SOURCE,
                "filename_pattern":
                    row.filename,
                "license":
                    row.license,
                "notes": (f"plot {row.plot_ids} ({row.plot_license}) "
                          "+ CC-BY 4.0 imagery"),
            })
    return pd.DataFrame(rules)


HEADER = """\
# Per-row license overrides, for sources whose annotations do not all share the
# license recorded in sources.csv.
#
# Each row narrows a source: every annotation whose `source` matches `source`
# (exactly, ignoring case and surrounding whitespace) AND whose `filename` matches
# the fnmatch `filename_pattern` takes `license` instead of the source-level one.
# Rows are applied top to bottom, so a later row wins over an earlier one; any
# annotation matched by no row keeps its source-level license.
#
# GENERATED FILE -- edit data_prep/ofo_license_overrides.py and rerun it, not this.
#
# Open Forest Observatory is the reason this file exists, and the only source in it.
# Its imagery and its annotations have different owners and different licenses:
#
#   * the image tile is cut from an OFO drone mission orthomosaic -- every mission in
#     the OFO catalog is CC-BY 4.0;
#   * the points are a contributor's ground-reference stem map, and those plots span
#     CC-BY 4.0, CC-BY-SA 4.0, CC-BY-NC-SA 4.0, CC0 1.0 and U.S. Forest Service
#     public domain.
#
# A packaged row ships both, so `license` below is the MOST RESTRICTIVE of the two,
# and `notes` records the plot license it came from. Plot licenses are the
# `license_short` column of ofo_ground-reference_plots.gpkg in Zenodo record
# https://zenodo.org/records/22731179.
#
# One mission can cover plots under different licenses, so a mission-wide pattern is
# followed by exact-filename rows for the tiles that differ.
#
# Lines starting with '#' are comments.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog-dir",
                        default=DEFAULT_CACHE_DIR,
                        help="Where to cache the Zenodo geopackages")
    parser.add_argument(
        "--validate",
        help=
        "Packaged release directory whose OFO filenames the rebuilt tiling must match"
    )
    parser.add_argument("--output", default=OVERRIDES_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    catalog = download_catalog(args.catalog_dir)
    check_mission_imagery_license(catalog[MISSIONS_GPKG])
    licenses_by_plot = plot_licenses(catalog[PLOTS_GPKG])

    tile_plots = tile_plot_table(licenses_by_plot)
    tiles = resolve_tiles(tile_plots)
    print(f"{len(tiles)} tiles across {tiles['mission_id'].nunique()} missions")
    print(tiles["plot_license"].value_counts().to_string())
    print()
    print(tiles["license"].value_counts().to_string())

    if args.validate:
        packaged = set()
        for path in glob.glob(os.path.join(args.validate, "*.csv")):
            table = pd.read_csv(path,
                                low_memory=False,
                                usecols=["filename", "source"])
            packaged |= set(table.loc[table["source"].astype(str).eq(SOURCE),
                                      "filename"])
        rebuilt = set(tiles["filename"])
        if packaged != rebuilt:
            raise SystemExit(
                f"Rebuilt tiling does not match {args.validate}: "
                f"{len(packaged - rebuilt)} packaged tiles unexplained, "
                f"{len(rebuilt - packaged)} rebuilt tiles not in the release.")
        print(
            f"\nvalidated: {len(packaged)} packaged OFO tiles all accounted for"
        )

    rules = build_rules(tiles)
    print(f"\n{len(rules)} override rules")
    if args.dry_run:
        print(rules.head(10).to_string())
        return 0

    with open(args.output, "w") as handle:
        handle.write(HEADER)
        rules.to_csv(handle, index=False)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
