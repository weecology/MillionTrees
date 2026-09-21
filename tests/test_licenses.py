"""Unit tests for the upstream-license lookup and row filter (no packaged data needed)."""
import pandas as pd
import pytest

from milliontrees.common import licenses as lic


def frame(rows):
    return pd.DataFrame(rows, columns=["source", "filename", "split"])


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("CC-BY 4.0", "CC-BY-4.0"),
    ("CC BY 4.0", "CC-BY-4.0"),
    ("cc_by_4", "CC-BY-4.0"),
    ("CC BY SA 4.0", "CC-BY-SA-4.0"),
    ("CC-BY-SA-4.0", "CC-BY-SA-4.0"),
    ("CC BY-NC-ND 3.0", "CC-BY-NC-ND-3.0"),
    ("CC0", "CC0-1.0"),
    ("CC0 1.0", "CC0-1.0"),
    ("CDLA-Permissive-1.0", "CDLA-Permissive-1.0"),
    ("AGPL-3.0 / GPL-3.0+", "AGPL-3.0-or-later"),
])
def test_manuscript_spellings_normalize(raw, expected):
    assert lic.normalize_license(raw) == expected


def test_unrecognized_license_raises():
    with pytest.raises(ValueError, match="Unrecognized license"):
        lic.normalize_license("MIT-ish")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def test_commercial_excludes_noncommercial_and_unknown():
    allowed = lic.resolve_selection("commercial")
    assert "CC-BY-4.0" in allowed
    assert "CC-BY-NC-4.0" not in allowed
    assert lic.UNKNOWN not in allowed


def test_permissive_excludes_copyleft_and_share_alike():
    allowed = lic.resolve_selection("permissive")
    assert "AGPL-3.0-or-later" not in allowed
    assert "CC-BY-NC-SA-4.0" not in allowed
    assert {"CC-BY-4.0", "CC0-1.0", "CDLA-Permissive-1.0"} <= allowed


def test_share_alike_is_commercial_but_not_permissive():
    """CC-BY-SA 4.0 arrived with the Open Forest Observatory plots: commercial use is
    fine, but a derivative has to carry the same terms."""
    assert "CC-BY-SA-4.0" in lic.resolve_selection("commercial")
    assert "CC-BY-SA-4.0" in lic.resolve_selection("derivatives")
    assert "CC-BY-SA-4.0" not in lic.resolve_selection("no-share-alike")
    assert "CC-BY-SA-4.0" not in lic.resolve_selection("permissive")
    assert "CC-BY-SA-4.0" in lic.resolve_selection("no-copyleft")


def test_selection_accepts_ids_patterns_and_unions():
    assert lic.resolve_selection("CC-BY-4.0") == {"CC-BY-4.0"}
    assert lic.resolve_selection("CC-BY-NC-*") == {
        "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-NC-ND-3.0"
    }
    union = lic.resolve_selection(["public-domain", "CC-BY-4.0"])
    assert union == {"CC0-1.0", "CC-BY-4.0"}


def test_none_selects_everything_including_unknown():
    assert lic.resolve_selection(None) == frozenset(lic.LICENSES)


def test_empty_pattern_match_raises():
    with pytest.raises(ValueError, match="matched nothing"):
        lic.resolve_selection("MIT-*")


# --------------------------------------------------------------------------- #
# Row resolution
# --------------------------------------------------------------------------- #
def test_source_names_match_case_and_whitespace_insensitively():
    df = frame([["  troles ET AL. 2024 ", "a.png", "train"]])
    assert lic.row_licenses(df).tolist() == ["CC-BY-NC-ND-3.0"]


def test_unlisted_source_is_unknown_and_warns():
    df = frame([["Not A Real Source", "a.png", "train"]])
    with pytest.warns(UserWarning, match="No license recorded"):
        assert lic.row_licenses(df).tolist() == [lic.UNKNOWN]


def test_overrides_refine_a_single_source_per_filename():
    """Open Forest Observatory is the case this exists for: one release source, one
    license per contributing plot, keyed on the mission id in the filename."""
    df = frame([
        ["OFO field 2025", "000018_ortho_616_OFO_field_2025.png", "train"],
        ["OFO field 2025", "000019_ortho_300_OFO_field_2025.png", "train"],
        ["Cloutier et al. 2023", "000018_something.png", "train"],
    ])
    overrides = [("ofo field 2025", "000018_*", "CC0-1.0")]
    resolved = lic.row_licenses(df, overrides=overrides)
    assert resolved.tolist() == [
        "CC0-1.0",  # overridden
        "CC-BY-NC-SA-4.0",  # source-level fallback
        "CC-BY-4.0",  # a different source, untouched by the pattern
    ]


def test_later_override_wins():
    df = frame([["OFO field 2025", "000018_a.png", "train"]])
    overrides = [("ofo field 2025", "000018_*", "CC0-1.0"),
                 ("ofo field 2025", "*_a.png", "CC-BY-4.0")]
    assert lic.row_licenses(df, overrides=overrides).tolist() == ["CC-BY-4.0"]


def test_mission_rule_then_per_tile_exception():
    """The shape the shipped OFO table uses: one rule for the mission, then exact
    filenames for the tiles inside it whose plot has different terms."""
    df = frame([
        ["OFO field 2025", "000091_ortho_10_OFO_field_2025.png", "train"],
        ["OFO field 2025", "000091_ortho_340_OFO_field_2025.png", "train"],
    ])
    overrides = [
        ("ofo field 2025", "000091_ortho_*_OFO_field_2025.png", "CC-BY-SA-4.0"),
        ("ofo field 2025", "000091_ortho_340_OFO_field_2025.png",
         "CC-BY-NC-SA-4.0"),
    ]
    assert lic.row_licenses(df, overrides=overrides).tolist() == [
        "CC-BY-SA-4.0", "CC-BY-NC-SA-4.0"
    ]


# --------------------------------------------------------------------------- #
# An explicit per-row license column
# --------------------------------------------------------------------------- #
def licensed_frame(rows):
    return pd.DataFrame(rows,
                        columns=["source", "filename", "split", "license"])


def test_explicit_license_column_beats_both_tables():
    df = licensed_frame([
        # More permissive than the source-level fallback.
        [
            "OFO field 2025", "000018_ortho_1_OFO_field_2025.png", "train",
            "CC-BY-4.0"
        ],
        # Stricter than the source-level license.
        ["Cloutier et al. 2023", "a.png", "train", "CC-BY-NC-4.0"],
    ])
    assert lic.row_licenses(df).tolist() == ["CC-BY-4.0", "CC-BY-NC-4.0"]


def test_explicit_license_column_splits_one_image():
    """The case a filename rule cannot express: one OFO tile straddling two field
    plots, so its own points carry two different licenses."""
    tile = "000133_ortho_589_OFO_field_2025.png"
    df = licensed_frame([
        ["OFO field 2025", tile, "train", "CC-BY-NC-SA-4.0"],
        ["OFO field 2025", tile, "train", "CC-BY-SA-4.0"],
    ])
    assert lic.row_licenses(df).tolist() == ["CC-BY-NC-SA-4.0", "CC-BY-SA-4.0"]
    kept, _ = lic.filter_by_license(df, "commercial")
    assert len(kept) == 1
    # The image is not dropped, it just carries one fewer annotation.
    assert kept["filename"].tolist() == [tile]


def test_blank_license_cells_fall_back_to_the_tables():
    """Only OFO rows will carry the column at first, so every other source arrives
    with it empty and must still resolve from sources.csv."""
    df = licensed_frame([
        ["Cloutier et al. 2023", "a.png", "train", None],
        ["Troles et al. 2024", "b.png", "train", ""],
        [
            "OFO field 2025", "000091_ortho_5_OFO_field_2025.png", "train",
            "CC-BY-SA-4.0"
        ],
    ])
    assert lic.row_licenses(df).tolist() == [
        "CC-BY-4.0", "CC-BY-NC-ND-3.0", "CC-BY-SA-4.0"
    ]


def test_explicit_license_column_is_normalized():
    df = licensed_frame(
        [["Cloutier et al. 2023", "a.png", "train", "CC BY SA 4.0"]])
    assert lic.row_licenses(df).tolist() == ["CC-BY-SA-4.0"]


def test_unrecognized_license_in_the_column_raises():
    df = licensed_frame([["Cloutier et al. 2023", "a.png", "train", "MIT-ish"]])
    with pytest.raises(ValueError, match="Unrecognized license"):
        lic.row_licenses(df)


def test_source_license_map_reports_explicit_licenses():
    df = licensed_frame([
        [
            "OFO field 2025", "000133_ortho_589_OFO_field_2025.png", "train",
            "CC-BY-NC-SA-4.0"
        ],
        [
            "OFO field 2025", "000133_ortho_589_OFO_field_2025.png", "train",
            "CC-BY-SA-4.0"
        ],
    ])
    assert "CC-BY-SA-4.0" in lic.source_license_map(df)["OFO field 2025"]


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def test_filter_drops_rows_without_touching_splits():
    df = frame([
        ["Cloutier et al. 2023", "a.png", "train"],
        ["Takeshige et al. 2025", "b.png", "train"],
        ["Takeshige et al. 2025", "c.png", "test"],
        ["Troles et al. 2024", "d.png", "test"],
    ])
    kept, _ = lic.filter_by_license(df, "commercial")
    assert kept["source"].tolist() == ["Cloutier et al. 2023"]
    # Surviving rows keep the split they arrived with.
    assert kept["split"].tolist() == ["train"]
    # The input frame is not mutated.
    assert len(df) == 4


def test_filter_none_is_a_no_op():
    df = frame([["Troles et al. 2024", "a.png", "train"]])
    kept, _ = lic.filter_by_license(df, None)
    assert kept.equals(df)


def test_filter_excludes_unknown_sources_by_default():
    df = frame([["World Resources Institute", "a.png", "train"],
                ["Cloutier et al. 2023", "b.png", "train"]])
    kept, _ = lic.filter_by_license(df, "commercial")
    assert kept["source"].tolist() == ["Cloutier et al. 2023"]
    kept_unknown, _ = lic.filter_by_license(df, ["commercial", "unknown"])
    assert len(kept_unknown) == 2


def test_filter_that_matches_nothing_raises():
    df = frame([["Troles et al. 2024", "a.png", "train"]])
    with pytest.raises(ValueError, match="matched no annotations"):
        lic.filter_by_license(df, "public-domain")


# --------------------------------------------------------------------------- #
# The shipped tables
# --------------------------------------------------------------------------- #
def test_shipped_tables_parse_and_use_known_licenses():
    table = lic.load_source_licenses()
    assert len(table) > 40
    assert set(table.values()) <= set(lic.LICENSES)
    for source_key, pattern, license_id in lic.load_license_overrides():
        assert license_id in lic.LICENSES
        assert source_key in table, (
            f"override for {source_key!r} has no source-level entry")
        assert pattern


def test_shipped_overrides_are_no_looser_than_their_source_fallback():
    """A source-level license is the fallback for any row no override matches, so it
    must be at least as restrictive as everything the overrides span -- otherwise a
    tile that slipped past the rules would be offered under terms it does not have."""
    table = lic.load_source_licenses()
    spans = {}
    for source_key, _pattern, license_id in lic.load_license_overrides():
        spans.setdefault(source_key, set()).add(license_id)
    for source_key, override_ids in spans.items():
        fallback = lic.LICENSES[table[source_key]]
        for override_id in override_ids:
            over = lic.LICENSES[override_id]
            assert fallback.commercial_use <= over.commercial_use
            assert fallback.derivatives <= over.derivatives
            assert fallback.share_alike >= over.share_alike
            assert fallback.copyleft >= over.copyleft


def test_shipped_ofo_overrides_cover_whole_missions():
    """Every OFO rule is scoped to one mission id, either mission-wide or to one of its
    tiles, and each mission-wide rule precedes its own exceptions."""
    rules = [
        (pattern, license_id)
        for source_key, pattern, license_id in lic.load_license_overrides()
        if source_key == "ofo field 2025"
    ]
    assert rules, "the shipped overrides table no longer carries OFO"
    seen_wide = set()
    for pattern, _license_id in rules:
        assert pattern.endswith("_OFO_field_2025.png")
        mission = pattern.split("_", 1)[0]
        assert mission.isdigit() and len(mission) == 6, pattern
        if "*" in pattern:
            assert mission not in seen_wide, f"two mission-wide rules for {mission}"
            seen_wide.add(mission)
        else:
            assert mission in seen_wide, (
                f"per-tile rule {pattern} has no mission-wide rule before it")


def test_every_shipped_source_name_is_unique():
    import pandas as pd
    raw = pd.read_csv(lic.SOURCES_PATH, comment="#")
    keys = raw["source"].map(lic._source_key)
    assert keys.is_unique, sorted(keys[keys.duplicated()])


# --------------------------------------------------------------------------- #
# Loader integration
# --------------------------------------------------------------------------- #
def _rewrite_fixture_sources(dataset_dir, sources):
    """Relabel the fixture split CSVs with real, licensed source names."""
    import glob
    import os
    for csv_path in glob.glob(os.path.join(dataset_dir, "*.csv")):
        df = pd.read_csv(csv_path)
        df["source"] = [sources[i % len(sources)] for i in range(len(df))]
        df.to_csv(csv_path, index=False)


@pytest.mark.parametrize("geometry,loader", [
    ("TreeBoxes", "milliontrees.datasets.TreeBoxes.TreeBoxesDataset"),
    ("TreePoints", "milliontrees.datasets.TreePoints.TreePointsDataset"),
    ("TreePolygons", "milliontrees.datasets.TreePolygons.TreePolygonsDataset"),
])
def test_loader_license_filter_keeps_splits_intact(dataset, geometry, loader,
                                                   tmp_path):
    """A license filter removes rows; it must not move anything between splits."""
    import importlib
    import os
    import shutil

    root = str(tmp_path)
    for name in (f"{geometry}_v0.0", f"{geometry}_supervised_v0.0"):
        shutil.copytree(os.path.join(dataset, name), os.path.join(root, name))
        _rewrite_fixture_sources(
            os.path.join(root, name),
            # One commercial-friendly source and one NonCommercial source.
            ["Cloutier et al. 2023", "Takeshige et al. 2025"])

    module_name, class_name = loader.rsplit(".", 1)
    klass = getattr(importlib.import_module(module_name), class_name)

    full = klass(root_dir=root, download=False, version="0.0", verbose=False)
    restricted = klass(root_dir=root,
                       download=False,
                       version="0.0",
                       verbose=False,
                       licenses="commercial")

    assert set(full.source_licenses) == {
        "Cloutier et al. 2023", "Takeshige et al. 2025"
    }
    assert restricted.source_licenses == {"Cloutier et al. 2023": ["CC-BY-4.0"]}
    assert len(restricted) < len(full)

    # Every image the filter keeps sits in exactly the split it had before.
    full_splits = dict(zip(full._input_array, full._split_array))
    restricted_splits = dict(
        zip(restricted._input_array, restricted._split_array))
    assert restricted_splits.items() <= full_splits.items()


def test_loader_default_does_not_filter(dataset):
    from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

    default = TreeBoxesDataset(root_dir=dataset,
                               download=False,
                               version="0.0",
                               verbose=False)
    explicit = TreeBoxesDataset(root_dir=dataset,
                                download=False,
                                version="0.0",
                                verbose=False,
                                licenses="all")
    assert default.licenses is None
    assert len(default) == len(explicit)
