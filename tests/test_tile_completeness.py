"""Unit tests for the per-tile completeness rule (no packaged data needed)."""
import numpy as np
import pandas as pd
import pytest

from milliontrees.common import tile_completeness as tc


def boxes_df(rows):
    return pd.DataFrame(rows, columns=["xmin", "ymin", "xmax", "ymax"])


def test_margin_zero_when_annotations_touch_every_edge():
    b = boxes_df([[0, 0, 50, 50], [950, 950, 1000, 1000]]).to_numpy(float)
    assert tc.max_margin(b, 1000, 1000) == 0.0


def test_margin_reports_widest_unannotated_strip():
    b = boxes_df([[800, 800, 1000, 1000]]).to_numpy(float)
    assert tc.max_margin(b, 1000, 1000) == pytest.approx(0.8)


def test_canopy_fraction_counts_only_canopy_inside_annotations():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, :50] = True                       # left half is canopy
    b = np.array([[0.0, 0.0, 25.0, 100.0]])   # annotation covers half of it
    assert tc.canopy_annotated_fraction(b, mask) == pytest.approx(0.5)


def test_edge_tile_is_rejected_and_interior_tile_accepted():
    interior = {"max_margin": 0.0, "canopy_annotated_frac": 0.99}
    edge = {"max_margin": 0.62, "canopy_annotated_frac": 0.16}
    assert tc.is_complete_tile(interior)
    assert not tc.is_complete_tile(edge)


def test_hole_in_the_middle_is_rejected_despite_zero_margin():
    """A tile can reach every edge and still leave a block of forest unlabelled."""
    stats = {"max_margin": 0.0, "canopy_annotated_frac": 0.2}
    assert not tc.is_complete_tile(stats)


def test_missing_canopy_falls_back_to_margin_test():
    assert tc.is_complete_tile({"max_margin": 0.0, "canopy_annotated_frac": float("nan")})
    assert not tc.is_complete_tile({"max_margin": 0.5, "canopy_annotated_frac": float("nan")})


def test_points_have_no_extent_so_canopy_test_is_undefined():
    pts = np.array([[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]])
    assert not tc.has_extent(pts)
    assert np.isnan(tc.canopy_annotated_fraction(pts, np.ones((50, 50), dtype=bool)))


def test_geometry_bounds_handles_boxes_and_points():
    assert tc.geometry_bounds(boxes_df([[1, 2, 3, 4]])).tolist() == [[1, 2, 3, 4]]
    pts = pd.DataFrame({"x": [5.0], "y": [6.0]})
    assert tc.geometry_bounds(pts).tolist() == [[5, 6, 5, 6]]


def test_manifest_returns_none_for_unknown_version(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame([{"version": "0.22", "split": "validation", "filename": "a.tif",
                   "source": "s", "n_annotations": 1, "max_margin": 0.0,
                   "canopy_annotated_frac": 1.0, "complete": True}]).to_csv(path, index=False)
    assert tc.manifest_complete_tiles("0.22", ["validation"], path) == {"a.tif"}
    assert tc.manifest_complete_tiles("0.99", ["validation"], path) is None
    assert tc.manifest_complete_tiles("0.22", ["test"], path) is None


# --------------------------------------------------------------------------- #
# Packaging step: the release must not contain incomplete validation tiles
# --------------------------------------------------------------------------- #
def _write_tile(image_dir, mask_dir, name, size=(100, 100), canopy=True):
    from PIL import Image
    Image.new("RGB", size).save(image_dir / name)
    mask = np.full((size[1], size[0]), 255 if canopy else 0, dtype=np.uint8)
    Image.fromarray(mask, mode="L").save(mask_dir / f"{name.rsplit('.', 1)[0]}.png")


def _validation_row(filename, orig_path, bounds, source="Allen et al. 2025"):
    x0, y0, x1, y1 = bounds
    return {"filename": filename, "orig_path": str(orig_path), "source": source,
            "existing_split": "validation", "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1}


@pytest.fixture
def packaging_tiles(tmp_path):
    """One interior tile (annotated edge to edge) and one edge tile (corner only)."""
    image_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    _write_tile(image_dir, mask_dir, "interior.png")
    _write_tile(image_dir, mask_dir, "edge.png")
    rows = [
        _validation_row("interior.png", image_dir / "interior.png", (0, 0, 100, 100)),
        _validation_row("edge.png", image_dir / "edge.png", (80, 80, 100, 100)),
    ]
    return tmp_path, mask_dir, pd.DataFrame(rows)


def test_packaging_drops_edge_tiles_from_every_geometry(packaging_tiles, tmp_path):
    from data_prep import package_datasets as pkg

    _, mask_dir, boxes = packaging_tiles
    points = boxes.drop(columns=["xmin", "ymin", "xmax", "ymax"]).assign(x=[50, 90], y=[50, 90])
    out = pkg.drop_incomplete_validation_tiles(
        {"TreeBoxes": boxes, "TreePoints": points, "TreePolygons": boxes.copy()},
        mask_dir, "v9.99", manifest_path=tmp_path / "manifest.csv")

    for name, df in out.items():
        assert set(df["filename"]) == {"interior.png"}, name
    # Points cannot be judged on their own; they must inherit the box verdict.
    assert set(out["TreePoints"]["filename"]) == set(out["TreeBoxes"]["filename"])


def test_packaging_leaves_non_validation_rows_alone(packaging_tiles, tmp_path):
    from data_prep import package_datasets as pkg

    _, mask_dir, boxes = packaging_tiles
    boxes = boxes.copy()
    boxes.loc[boxes["filename"] == "edge.png", "existing_split"] = "train"
    out = pkg.drop_incomplete_validation_tiles(
        {"TreeBoxes": boxes}, mask_dir, "v9.99", manifest_path=tmp_path / "manifest.csv")
    assert set(out["TreeBoxes"]["filename"]) == {"interior.png", "edge.png"}


def test_packaging_writes_the_manifest(packaging_tiles, tmp_path):
    from data_prep import package_datasets as pkg

    _, mask_dir, boxes = packaging_tiles
    manifest = tmp_path / "manifest.csv"
    pkg.drop_incomplete_validation_tiles({"TreeBoxes": boxes}, mask_dir, "v9.99",
                                         manifest_path=manifest)
    written = pd.read_csv(manifest, dtype={"version": str})
    assert set(written["version"]) == {"9.99"}
    assert set(written["split"]) == {"validation"}
    assert dict(zip(written["filename"], written["complete"])) == {
        "interior.png": True, "edge.png": False}
