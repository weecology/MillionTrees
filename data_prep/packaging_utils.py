"""Shared helpers for source-unique packaged filenames.

Image basenames are NOT unique across MillionTrees source datasets (many sources
reuse generic names like ``144.jpg``), and a few sources even reuse a basename
across sub-folders (e.g. Santos 2019 ``train/img/X.JPG`` and ``test/img/X.JPG``).
Packaging and tree-coverage-mask precomputation both key files by basename/stem,
so collisions silently mispair images with the wrong source's mask (and merge
annotations across sources).

To make every packaged file unique *and* self-describing, packaged filenames
embed the source: ``<stem>_<sanitized_source><ext>``. When that is still not
unique (same stem+source from different paths) a short, deterministic hash of
the original path is appended. ``package_datasets.py`` and
``precompute_tree_coverage_masks.py`` both build the name map from the same
annotation-CSV union, so they agree on every key.
"""
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def sanitize_source(source: str) -> str:
    """Turn a human source label into a filename-safe token.

    ``'Velasquez-Camacho et al. 2023'`` -> ``'Velasquez_Camacho_et_al_2023'``.
    """
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(source)).strip("_")
    return token or "unknown_source"


def _short_hash(text: str, length: int = 6) -> str:
    return hashlib.blake2s(text.encode("utf-8"), digest_size=8).hexdigest()[:length]


def _resolve_image_path(row_path: str, root_dir: str | None) -> str:
    """Mirror filter_auto_arborist_tcd.resolve_image_path without heavy imports."""
    if os.path.isabs(row_path) and os.path.exists(row_path):
        return row_path
    if root_dir and not os.path.isabs(row_path):
        candidate = os.path.join(root_dir, os.path.basename(row_path))
        if os.path.exists(candidate):
            return candidate
    return row_path


def collect_image_source_pairs(annotation_csvs, root_dir=None):
    """Return sorted unique ``(resolved_path, source)`` pairs across all CSVs."""
    pairs = {}
    for csv_path in annotation_csvs:
        df = pd.read_csv(csv_path, low_memory=False)
        col = "filename" if "filename" in df.columns else "image_path"
        if col not in df.columns:
            raise ValueError(f"{csv_path} must include 'filename' or 'image_path'")
        if "source" in df.columns:
            sources = df["source"].astype(str)
        else:
            fallback = Path(csv_path).parent.name
            print(f"WARNING: {csv_path} has no 'source' column; using '{fallback}'")
            sources = [fallback] * len(df)
        for raw_path, source in zip(df[col], sources):
            if pd.isna(raw_path):
                continue
            pairs[(_resolve_image_path(str(raw_path), root_dir), str(source))] = None
    return sorted(pairs.keys())


def build_unique_name_map(pairs):
    """Map ``orig_path -> packaged_filename`` for a set of ``(path, source)`` pairs.

    The packaged name is ``<stem>_<source><ext>``. Masks are keyed by stem, so
    uniqueness is enforced at the stem level: when two distinct images would
    share a stem (same basename+source from different paths, or same basename
    with different extensions), a short hash of the original path is appended to
    every member of the colliding group. Deterministic given the same pair set.
    """
    by_stem = defaultdict(list)  # base packaged stem -> [(orig_path, suffix)]
    for orig_path, source in pairs:
        p = Path(orig_path)
        base_stem = f"{p.stem}_{sanitize_source(source)}"
        by_stem[base_stem].append((orig_path, p.suffix))

    name_map = {}
    for base_stem, members in by_stem.items():
        distinct_paths = {orig for orig, _ in members}
        collide = len(distinct_paths) > 1
        for orig_path, suffix in members:
            if collide:
                name_map[orig_path] = f"{base_stem}_{_short_hash(orig_path)}{suffix}"
            else:
                name_map[orig_path] = f"{base_stem}{suffix}"
    return name_map
