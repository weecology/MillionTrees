"""Regression cover for train/test leaks in the packaged splits (no data needed).

An earlier implementation of the within-distribution auto-split picked the ~10%
test fraction by slicing annotation-*row* index instead of grouping by image
filename first. Whichever image's rows straddled the cut landed partly in train
and partly in test -- confirmed on the packaged v0.23 CSVs (13 TreeBoxes / 7
TreePoints / 22 TreePolygons images split across both splits). See
notes/prerelease_v1_packaging_audit.md (findings #1/#2).
"""
import os

import pandas as pd
import pytest

from data_prep import package_datasets as pkg


def _many_rows_per_image(n_images_per_source=10, rows_per_image=7, sources=("Source A", "Source B")):
    """Build a multi-row-per-image annotation frame, mimicking one box/point/
    polygon per tree with several trees per image -- the shape that exposed the
    row-slicing bug (a single image's rows are far more likely to straddle a
    row-index cut than a filename-index cut)."""
    rows = []
    for source in sources:
        for i in range(n_images_per_source):
            filename = f"{source}_{i}.tif"
            for _ in range(rows_per_image):
                rows.append({"filename": filename, "source": source})
    return pd.DataFrame(rows)


def test_within_distribution_auto_split_keeps_each_image_on_one_side():
    df = _many_rows_per_image()
    out = pkg.assign_within_distribution_auto_split(df)

    assert out["split"].isin(["train", "test"]).all()

    per_image_splits = out.groupby("filename")["split"].nunique()
    leaked = per_image_splits[per_image_splits > 1]
    assert leaked.empty, f"images split across train/test: {leaked.index.tolist()}"


def test_within_distribution_auto_split_holds_out_roughly_ten_percent_of_images():
    df = _many_rows_per_image(n_images_per_source=20, rows_per_image=5, sources=("Source A",))
    out = pkg.assign_within_distribution_auto_split(df)

    n_images = out["filename"].nunique()
    n_test_images = out.loc[out["split"] == "test", "filename"].nunique()
    assert n_test_images == round(n_images * 0.1)


def test_within_distribution_auto_split_respects_existing_split_pins():
    df = _many_rows_per_image(n_images_per_source=5, rows_per_image=3, sources=("Pinned Source",))
    df["existing_split"] = "train"
    out = pkg.assign_within_distribution_auto_split(df)
    assert (out["split"] == "train").all()


@pytest.mark.parametrize("test_sources,train_sources", [
    (["Held Out A", "Held Out B"], ["Kept In"]),
])
def test_ood_assignment_is_all_or_nothing_per_source(test_sources, train_sources):
    """_assign_ood_split_by_source must never split a single source across
    train and test -- unlike within-distribution, OOD membership is a
    source-level decision (see notes/ood_split_test_sources_and_leaks.md)."""
    df = _many_rows_per_image(n_images_per_source=4, rows_per_image=3,
                              sources=test_sources + train_sources)
    df["split"] = "train"
    out = pkg._assign_ood_split_by_source(df, test_sources, train_sources)

    for source in test_sources:
        assert (out.loc[out["source"] == source, "split"] == "test").all()
    for source in train_sources:
        assert (out.loc[out["source"] == source, "split"] == "train").all()


def test_no_filename_shared_across_train_and_test_end_to_end(tmp_path):
    """End-to-end: run the real within_distribution_split I/O function and check
    the CSV it writes on disk, not just the in-memory helper."""
    df = _many_rows_per_image(n_images_per_source=15, rows_per_image=6,
                              sources=("Alpha et al.", "Beta et al.", "Gamma et al."))
    version = "v0.0"
    for geometry in ("TreePolygons", "TreePoints", "TreeBoxes"):
        os.makedirs(f"{tmp_path}/Test{geometry}_{version}", exist_ok=True)
    pkg.within_distribution_split(df.copy(), df.copy(), df.copy(),
                                  base_dir=f"{tmp_path}/", version=version,
                                  prefix="Test")

    for geometry in ("TreePolygons", "TreePoints", "TreeBoxes"):
        out = pd.read_csv(f"{tmp_path}/Test{geometry}_{version}/within-distribution.csv")
        per_image_splits = out.groupby("filename")["split"].nunique()
        leaked = per_image_splits[per_image_splits > 1]
        assert leaked.empty, f"{geometry}: images split across train/test: {leaked.index.tolist()}"
