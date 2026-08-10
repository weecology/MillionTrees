"""Generate ``data_prep/tile_completeness.csv``: which eval tiles are annotated wall to wall.

The packaged ``complete`` flag is per source (``source_completeness.csv``). That is too coarse
for the held-out TLS validation sources, which mix interior tiles annotated edge to edge with
**edge tiles** where the plot footprint clips a corner and the rest of the tile is unlabelled
forest. AP charges every detection in the unlabelled part as a false positive, so an edge tile
scores the annotation protocol rather than the model (``notes/validation_ap_completeness.md``).

This script applies ``milliontrees.common.tile_completeness`` to a packaged release and writes
one row per tile. The loaders read the result when called with ``complete_tiles_only=True``.

The flag is derived from a geometry with *extent* (boxes by default, polygons on request) and
keyed by filename, so the point task -- whose stem points sit half a crown inside the tile edge
and cover no canopy pixels -- inherits the same decision for the same image instead of being
judged by a rule that cannot work on centroids.

Rerun this after every re-tiling or mask regeneration; the tile geometry changes with both.

Usage:
    python data_prep/flag_tile_completeness.py --version 0.22 --splits validation
    python data_prep/flag_tile_completeness.py --version 0.22 --splits validation --dry-run
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from milliontrees.common.tile_completeness import (DEFAULT_MAX_MARGIN,
                                                   DEFAULT_MIN_CANOPY_ANNOTATED,
                                                   MANIFEST_COLUMNS, MANIFEST_PATH,
                                                   is_complete_tile, score_tiles)

GEOMETRY_PACKAGE = {"boxes": "TreeBoxes", "polygons": "TreePolygons"}


def read_split_rows(csv_path, splits, chunksize=500_000):
    """Rows for ``splits`` from a packaged split CSV, read in chunks (these are ~2 GB)."""
    kept = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        kept.append(chunk[chunk["split"].isin(splits)])
    return pd.concat(kept, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root-dir", default=os.environ.get(
        "MT_ROOT", "/orange/ewhite/web/public/MillionTrees"))
    ap.add_argument("--version", required=True, help="Release version, e.g. 0.22")
    ap.add_argument("--splits", nargs="+", default=["validation"],
                    help="Splits to score (default: validation).")
    ap.add_argument("--geometry", default="boxes", choices=sorted(GEOMETRY_PACKAGE),
                    help="Which packaged geometry supplies the extent (default: boxes).")
    ap.add_argument("--split-scheme", default="out-of-distribution",
                    help="Any scheme works; validation rows are identical across schemes.")
    ap.add_argument("--unsupervised-package", action="store_true",
                    help="Read the full package instead of the _supervised_ one the loaders "
                         "default to.")
    ap.add_argument("--max-margin", type=float, default=DEFAULT_MAX_MARGIN)
    ap.add_argument("--min-canopy-annotated", type=float, default=DEFAULT_MIN_CANOPY_ANNOTATED)
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the summary without writing the manifest.")
    args = ap.parse_args()

    suffix = "" if args.unsupervised_package else "_supervised"
    data_dir = Path(args.root_dir) / f"{GEOMETRY_PACKAGE[args.geometry]}{suffix}_v{args.version}"
    csv_path = data_dir / f"{args.split_scheme}.csv"
    if not csv_path.exists():
        raise SystemExit(f"No packaged split CSV at {csv_path}")

    print(f"Scoring {args.splits} tiles from {data_dir.name} ({args.geometry} geometry)")
    df = read_split_rows(csv_path, args.splits)
    if df.empty:
        raise SystemExit(f"No rows for splits {args.splits} in {csv_path}")

    split_by_file = df.drop_duplicates("filename").set_index("filename")["split"]
    stats = score_tiles(df, data_dir, progress=True)
    stats["split"] = stats["filename"].map(split_by_file)
    stats["version"] = str(args.version)
    stats["complete"] = stats.apply(
        lambda r: is_complete_tile(r, args.max_margin, args.min_canopy_annotated), axis=1)

    print(f"\n{'source':<24}{'tiles':>7}{'complete':>10}{'dropped':>9}")
    print("-" * 50)
    for source, group in stats.groupby("source"):
        n_complete = int(group["complete"].sum())
        print(f"{source:<24}{len(group):>7}{n_complete:>10}{len(group) - n_complete:>9}")
    print("-" * 50)
    print(f"{'TOTAL':<24}{len(stats):>7}{int(stats['complete'].sum()):>10}"
          f"{int((~stats['complete']).sum()):>9}")

    rows = stats[MANIFEST_COLUMNS]
    if args.dry_run:
        print("\n--dry-run: manifest not written")
        return

    manifest_path = Path(args.manifest)
    existing = None
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, dtype={"version": str})
        # Replace only the version/split combinations this run regenerated, so a manifest
        # covering several releases keeps its other rows.
        drop = (existing["version"] == str(args.version)) & existing["split"].isin(args.splits)
        existing = existing[~drop]
    out = pd.concat([existing, rows], ignore_index=True) if existing is not None else rows
    out = out.sort_values(["version", "split", "source", "filename"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(manifest_path, index=False)
    print(f"\nWrote {manifest_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
