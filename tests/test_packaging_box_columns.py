"""Unit tests for box/geometry reconciliation during packaging (no data needed).

Regression cover for the silent drop that removed six configured TreeBoxes
sources from v0.20-v0.22: sources ship either explicit box columns or a geometry
WKT column, and whichever side a source omitted used to be filtered away.
"""
import pandas as pd
import pytest

from data_prep import package_datasets as pkg


def _boxes_only(source="Boxes Only et al."):
    return pd.DataFrame({
        "xmin": [10.0, 30.0], "ymin": [20.0, 40.0],
        "xmax": [50.0, 70.0], "ymax": [60.0, 80.0],
        "filename": ["a.tif", "a.tif"], "source": source,
    })


def _geometry_only(source="Geometry Only et al."):
    return pd.DataFrame({
        "geometry": ["POLYGON ((10 20, 50 20, 50 60, 10 60, 10 20))",
                     "POLYGON ((30 40, 70 40, 70 80, 30 80, 30 40))"],
        "filename": ["b.tif", "b.tif"], "source": source,
    })


def _combined():
    return pd.concat([_boxes_only(), _geometry_only()], ignore_index=True)


def test_both_representations_survive_the_packaging_filters():
    df = pkg.reconcile_box_columns(_combined(), "TreeBoxes")
    df = pkg.filter_invalid_boxes(df, "TreeBoxes")
    df = pkg.process_geometry_columns(df, "box")
    df = pkg.filter_degenerate_boxes(df)

    assert df["source"].value_counts().to_dict() == {
        "Boxes Only et al.": 2, "Geometry Only et al.": 2}


def test_geometry_is_derived_from_box_columns():
    df = pkg.reconcile_box_columns(_boxes_only(), "TreeBoxes")
    assert df["geometry"].notna().all()
    bounds = pkg._parse_geometry_column(df["geometry"]).bounds
    assert bounds["minx"].tolist() == [10.0, 30.0]
    assert bounds["maxy"].tolist() == [60.0, 80.0]


def test_box_columns_are_derived_from_geometry():
    df = pkg.reconcile_box_columns(_geometry_only(), "TreeBoxes")
    assert df[pkg.BOX_COLUMNS].notna().all().all()
    assert df["xmin"].tolist() == [10.0, 30.0]
    assert df["ymax"].tolist() == [60.0, 80.0]


def test_geometry_wins_over_stale_box_columns():
    df = _geometry_only()
    df[pkg.BOX_COLUMNS] = [[1.0, 2.0, 3.0, 4.0]] * 2
    out = pkg.reconcile_box_columns(df, "TreeBoxes")
    # reconcile leaves both alone; process_geometry_columns re-derives from geometry
    out = pkg.process_geometry_columns(out, "box")
    assert out["xmin"].tolist() == [10.0, 30.0]


def test_genuinely_unusable_boxes_are_still_dropped():
    df = _boxes_only()
    df.loc[0, "xmax"] = df.loc[0, "xmin"]  # zero width
    df.loc[1, "ymin"] = None               # unusable, and no geometry to recover it
    out = pkg.filter_invalid_boxes(pkg.reconcile_box_columns(df, "TreeBoxes"), "TreeBoxes")
    assert len(out) == 0


def test_dropped_sources_are_reported(capsys):
    df = _boxes_only()
    df["xmax"] = df["xmin"]
    pkg.filter_invalid_boxes(df, "TreeBoxes")
    out = capsys.readouterr().out
    assert "Boxes Only et al." in out
    assert "SOURCE REMOVED ENTIRELY" in out


@pytest.mark.parametrize("missing", ["xmin", "geometry"])
def test_reconcile_tolerates_a_column_absent_from_every_source(missing):
    df = _combined().drop(columns=[missing])
    out = pkg.reconcile_box_columns(df, "TreeBoxes")
    assert set(pkg.BOX_COLUMNS + ["geometry"]).issubset(out.columns)
