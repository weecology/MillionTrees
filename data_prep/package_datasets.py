import pandas as pd
import os
import shutil
import geopandas as gpd
import zipfile
from deepforest.visualize import plot_results
from deepforest.utilities import read_file
import cv2
import rasterio
import glob
from pathlib import Path
from shapely.geometry.base import BaseGeometry
try:
    from data_prep.packaging_utils import (
        build_unique_name_map,
        collect_image_source_pairs,
    )
except ImportError:  # when run as a script from inside data_prep/
    from packaging_utils import build_unique_name_map, collect_image_source_pairs
import shapely.wkt

# Zero-shot test sources shared between zero_shot_split and cross_geometry_split.
# Cross-geometry uses the polygon test list as its evaluation set so the two
# splits stay aligned.
ZEROSHOT_TEST_SOURCES_POLYGONS = [
    "Troles et al. 2024",
    "Bolhman 2008",
    "Lefebvre et al. 2024",
    "NEON MultiTemporal",
    "Takeshige et al. 2025",
    "Alejandro_Miranda",
]
ZEROSHOT_TEST_SOURCES_POINTS = [
    "Amirkolaee et al. 2023",
    "NEON_points",
    "NEON MultiTemporal",
    "OSBS megaplot 2025",
    "OFO field 2025",
]
ZEROSHOT_TEST_SOURCES_BOXES = [
    "Radogoshi et al. 2021",
    "SelvaBox",
    "NEON_benchmark",
]

# Canonical token that marks a source as unsupervised/weakly-labeled. The data
# loader excludes these by default via exclude_sources=['*unsupervised*'].
UNSUPERVISED_SUFFIX = "unsupervised"

# Sources that are unsupervised/weakly-labeled but whose upstream annotation CSVs
# can ship WITHOUT the canonical suffix (e.g. SPREAD's "Feng et al. 2025" when
# data_prep/SPREAD.py has not been re-run). normalize_unsupervised_sources()
# rewrites these to "<name> unsupervised" at packaging time so the label is
# permanent in every released CSV and the loader's '*unsupervised*' exclusion
# always catches them. This is the single source of truth — add a source here
# instead of sprinkling per-source masks through the splitting code.
UNSUPERVISED_SOURCE_ALIASES = {
    "Feng et al. 2025",
}


def normalize_unsupervised_sources(datasets):
    """Force every known-unsupervised source name to carry the canonical suffix.

    Idempotent: a source that already contains 'unsupervised' or 'weak supervised'
    (case-insensitive) is left unchanged; declared aliases get ' unsupervised'
    appended exactly once. Run this immediately after loading/combining so all
    downstream splitting, filtering, and the published CSVs see the correct name.
    """
    if "source" not in datasets.columns:
        return datasets
    src = datasets["source"].astype(str)
    already = src.str.contains("unsupervised|weak supervised", case=False, na=False)
    alias = src.isin(UNSUPERVISED_SOURCE_ALIASES) & ~already
    if alias.any():
        renamed = sorted(src[alias].unique())
        datasets.loc[alias, "source"] = src[alias] + f" {UNSUPERVISED_SUFFIX}"
        print(f"Relabeled unsupervised sources (added '{UNSUPERVISED_SUFFIX}'): {renamed}")
    return datasets


# Canonical full names of every unsupervised/weakly-labeled source we ship, per
# geometry. verify_unsupervised_sources() checks these survived packaging: it
# fails loudly if a base name shows up WITHOUT the 'unsupervised' suffix (the
# SPREAD/Feng regression). Unlike UNSUPERVISED_SOURCE_ALIASES it does NOT rename
# anything, so listing a source whose base also has a supervised variant (e.g.
# the NEON 'Weinstein et al. 2018' box benchmark) is safe.
UNSUPERVISED_SOURCES_EXPECTED = {
    "TreePolygons": ["Feng et al. 2025 unsupervised"],
    "TreePoints": ["Young et al. 2025 unsupervised"],
    "TreeBoxes": ["Weinstein et al. 2018 unsupervised"],
}


def verify_unsupervised_sources(datasets, dataset_type):
    """Verify expected unsupervised sources kept their canonical suffix.

    Turns a silent labeling regression (an unsupervised source shipping without
    the 'unsupervised' suffix, as SPREAD/Feng did) into a hard failure at
    packaging time. Does not modify the data.
    """
    expected = UNSUPERVISED_SOURCES_EXPECTED.get(dataset_type, [])
    if not expected or "source" not in datasets.columns:
        return
    present = set(datasets["source"].astype(str).unique())
    present_lower = {s.lower() for s in present}
    problems = []
    for canonical in expected:
        base = canonical.lower()
        if base.endswith(UNSUPERVISED_SUFFIX):
            base = base[: -len(UNSUPERVISED_SUFFIX)].strip()
        # The bare (un-suffixed) base name must not appear as its own source.
        if base in present_lower:
            problems.append(
                f"{base!r} is present without the '{UNSUPERVISED_SUFFIX}' suffix "
                f"(expected {canonical!r}) -- add it to UNSUPERVISED_SOURCE_ALIASES"
            )
        elif canonical not in present:
            # Not fatal: a sub-release (Mini/Small) may legitimately drop it.
            print(f"Note: expected unsupervised source {canonical!r} not present in {dataset_type}")
    if problems:
        raise ValueError(
            f"{dataset_type}: unsupervised labeling regression -> " + "; ".join(problems)
        )

def remove_alpha_channel(datasets):
    """Remove alpha channels from images in the dataset."""
    for source in datasets["source"].unique():
        source_images = datasets[datasets["source"] == source]["filename"].unique()
        for image in source_images:
            with rasterio.open(image) as img_src:
                if img_src.count == 4:  # Check if the image has 4 channels
                    data = img_src.read([1, 2, 3])  # Read only the first three channels
                    profile = img_src.profile
                    profile.update(count=3)  # Update profile to reflect 3 channels
                    new_image_path = os.path.splitext(image)[0] + "_no_alpha.tif"
                    with rasterio.open(new_image_path, 'w', **profile) as img_dst:
                        img_dst.write(data)
                    datasets.loc[datasets["filename"] == image, "filename"] = new_image_path


SOURCE_COMPLETENESS_CSV = Path(__file__).parent / "source_completeness.csv"


def load_source_completeness(path=SOURCE_COMPLETENESS_CSV):
    """Load source -> complete mapping from the completeness CSV.

    Returns an empty dict if the file is missing so the pipeline still runs;
    sources absent from the CSV default to ``complete=False`` downstream.
    """
    if not Path(path).exists():
        print(f"Warning: source_completeness.csv not found at {path}")
        return {}
    df = pd.read_csv(path)
    # Normalize bool-like strings ("True"/"False") into booleans
    df["complete"] = df["complete"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    if df["complete"].isna().any():
        bad = df.loc[df["complete"].isna(), "source"].tolist()
        raise ValueError(
            f"source_completeness.csv has unparseable complete values for: {bad}"
        )
    return dict(zip(df["source"].str.strip(), df["complete"]))


def combine_datasets(dataset_paths, debug=False):
    """Concatenate per-source annotation CSVs and tag each row with completeness.

    ``complete`` is set per source from ``source_completeness.csv``. Sources
    missing from that file default to ``False`` and a warning is printed once
    per missing source. The ``complete`` flag drives the counting MAE metric:
    ``CountingError`` is computed only over sources flagged ``True`` (i.e.
    exhaustively annotated).
    """
    datasets = []
    for dataset_path in dataset_paths:
        df = pd.read_csv(dataset_path, low_memory=False)
        if "image_path" in df.columns:
            if "filename" in df.columns:
                df = df.drop(columns="filename")
            df = df.rename(columns={"image_path": "filename"})
            df.reset_index(drop=True, inplace=True)
        datasets.append(df)

    combined_df = pd.concat(datasets, ignore_index=True)

    completeness_map = load_source_completeness()
    combined_df["complete"] = combined_df["source"].map(completeness_map)
    missing_sources = sorted(
        set(combined_df.loc[combined_df["complete"].isna(), "source"].unique())
    )
    if missing_sources:
        print(
            "Warning: sources missing from source_completeness.csv "
            f"(defaulting complete=False): {missing_sources}"
        )
        combined_df["complete"] = combined_df["complete"].fillna(False)
    combined_df["complete"] = combined_df["complete"].astype(bool)

    return combined_df


def split_dataset(datasets, split_column="filename", frac=0.8):
    """Split the dataset into training and testing sets."""
    train_images = datasets[split_column].drop_duplicates().sample(frac=frac)
    datasets.loc[datasets[split_column].isin(train_images), "split"] = "train"
    datasets.loc[~datasets[split_column].isin(train_images), "split"] = "test"
    
    return datasets

def keep_columns_if_exist(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Return df with only the columns from cols that actually exist in df (preserve order)."""
        existing = [c for c in cols if c in df.columns]
        return df[existing]

def process_geometry_columns(datasets, geom_type):
    """Process geometry columns based on the dataset type."""
    # Filter out rows with None geometries before processing
    if "geometry" in datasets.columns:
        datasets = datasets[datasets["geometry"].notna()].copy()
    
    if len(datasets) == 0:
        return datasets
    
    # Build a GeoSeries from either WKT strings or shapely geometries
    def convert_to_shapely(value):
        if value is None:
            return None
        elif type(value) == str:
            return shapely.wkt.loads(value)
        elif isinstance(value, BaseGeometry):
            return value
        else:
            raise ValueError(f"Invalid geometry type: {type(value)}")
    shapely_geometries = gpd.GeoSeries(datasets["geometry"].apply(convert_to_shapely))
    
    if geom_type == "box":
        bounds = shapely_geometries.bounds
        datasets[["xmin", "ymin", "xmax", "ymax"]] = bounds[["minx", "miny", "maxx", "maxy"]].to_numpy()
        datasets["geometry"] = shapely_geometries.to_wkt()
    elif geom_type == "point":
        centroids = shapely_geometries.centroid
        datasets["x"] = centroids.x
        datasets["y"] = centroids.y
        datasets["geometry"] = shapely_geometries.to_wkt()
    elif geom_type == "polygon":
        # Keep only simple Polygons — MultiPolygon, GeometryCollection, and other
        # compound types lack .exterior and will crash the dataloader mask renderer.
        mask = shapely_geometries.geom_type == "Polygon"
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            dropped_types = shapely_geometries[~mask].geom_type.value_counts().to_dict()
            print(f"Filtered out {n_dropped} non-Polygon geometries: {dropped_types}")
        datasets = datasets.loc[mask].copy()
        shapely_geometries = shapely_geometries.loc[mask].copy()
        datasets["polygon"] = shapely_geometries.to_wkt()
        datasets["geometry"] = shapely_geometries.to_wkt()
    
    return datasets


def filter_degenerate_boxes(datasets):
    """Filter out boxes with invalid or zero/negative extent.

    This operates in *source coordinates* before any train/val split so
    degenerate boxes never enter the packaged dataset at all.
    """
    if "xmin" not in datasets.columns or "xmax" not in datasets.columns:
        return datasets
    for col in ("xmin", "ymin", "xmax", "ymax"):
        datasets[col] = pd.to_numeric(datasets[col], errors="coerce")
    # Basic geometry sanity: strictly positive width/height
    valid_mask = (datasets["xmax"] > datasets["xmin"]) & (datasets["ymax"] > datasets["ymin"])
    # Drop any rows with NaNs in the box columns as well
    valid_mask &= datasets[["xmin", "ymin", "xmax", "ymax"]].notna().all(axis=1)
    n_filtered = len(datasets) - valid_mask.sum()
    if n_filtered > 0:
        print(f"Filtered out {n_filtered} degenerate boxes (zero width or height)")
    return datasets.loc[valid_mask].copy()


def filter_degenerate_polygons(datasets):
    """Filter out polygons that would create invalid bounding boxes (zero width or height)."""
    def convert_to_shapely(value):
        if type(value) == str:
            return shapely.wkt.loads(value)
        elif isinstance(value, BaseGeometry):
            return value
        else:
            raise ValueError(f"Invalid geometry type: {type(value)}")
    
    shapely_geometries = gpd.GeoSeries(datasets["polygon"].apply(convert_to_shapely))
    bounds = shapely_geometries.bounds
    
    # Filter out polygons where xmin >= xmax or ymin >= ymax
    # Using >= instead of != to catch cases where max < min due to precision issues
    # Also filter polygons that are extremely small (< 1 pixel in either dimension)
    # as they will create degenerate masks when rounded to integer coordinates
    valid_mask = (bounds["maxx"] > bounds["minx"]) & (bounds["maxy"] > bounds["miny"])
    
    # Additional check: filter polygons smaller than 1 pixel (would round to zero width/height)
    # This is a conservative check - we assume images are at least 100x100 pixels
    # In practice, degenerate masks can still occur, so filtering also happens in __getitem__
    width = bounds["maxx"] - bounds["minx"]
    height = bounds["maxy"] - bounds["miny"]
    size_mask = (width >= 1.0) & (height >= 1.0)
    valid_mask = valid_mask & size_mask
    
    n_filtered = len(datasets) - valid_mask.sum()
    if n_filtered > 0:
        print(f"Filtered out {n_filtered} degenerate polygons (zero width or height)")
    
    return datasets.loc[valid_mask].copy()


def drop_duplicate_annotations(datasets, label=""):
    """Drop exact duplicate annotations (same image + geometry + source).

    Upstream prediction CSVs have shipped the same box/point/polygon multiple
    times (differing only in columns the packager drops, such as prediction
    confidence), which collapses into identical published rows and inflates
    annotation counts. This removes those duplicates on the canonical identity.
    """
    key = [c for c in ("filename", "geometry", "source") if c in datasets.columns]
    if not key or "geometry" not in datasets.columns:
        return datasets
    n_before = len(datasets)
    datasets = datasets.drop_duplicates(subset=key).copy()
    n_dropped = n_before - len(datasets)
    if n_dropped > 0:
        print(f"{label}: dropped {n_dropped} duplicate annotations "
              f"({n_before} -> {len(datasets)})")
    return datasets


def create_directories(base_dir, dataset_type):
    """Create directories for the dataset."""
    os.makedirs(f"{base_dir}{dataset_type}_{version}/images", exist_ok=True)
    os.makedirs(f"{base_dir}{dataset_type}_{version}/masks", exist_ok=True)
    os.makedirs(f"{base_dir}Mini{dataset_type}_{version}/images", exist_ok=True)
    os.makedirs(f"{base_dir}Mini{dataset_type}_{version}/masks", exist_ok=True)
    os.makedirs(f"{base_dir}Small{dataset_type}_{version}/images", exist_ok=True)
    os.makedirs(f"{base_dir}Small{dataset_type}_{version}/masks", exist_ok=True)


def assign_packaged_filenames(datasets, name_map):
    """Rename images to a source-unique packaged name (see packaging_utils).

    Image basenames are not unique across sources, which silently mispairs
    images with the wrong source's tree-coverage mask (and merges annotations
    across sources). Store the original source path in ``orig_path`` (used to
    copy the image/mask) and overwrite ``filename`` with the packaged name from
    ``name_map`` (built from the same CSV union as the mask precompute, so the
    keys agree). Raises if any image path is missing from the map, or if the
    result is not unique at the stem level (masks are keyed by stem).
    """
    datasets = datasets.copy()
    datasets["orig_path"] = datasets["filename"]
    missing = sorted(set(datasets["orig_path"]) - set(name_map))
    if missing:
        raise ValueError(
            f"{len(missing)} image path(s) absent from the packaged name map, "
            f"e.g. {missing[:3]}. Is the annotation_csvs.cfg union consistent?"
        )
    datasets["filename"] = datasets["orig_path"].map(name_map)
    stems = datasets["filename"].map(lambda f: Path(f).stem)
    distinct_sources = datasets.assign(_stem=stems).groupby("_stem")["orig_path"].nunique()
    collisions = distinct_sources[distinct_sources > 1]
    if len(collisions):
        raise ValueError(
            f"Packaged filename collision: {len(collisions)} stem(s) map to >1 "
            f"source image, e.g. {collisions.index[:5].tolist()}."
        )
    return datasets


def copy_images(datasets, base_dir, dataset_type):
    """Copy each source image to the package under its source-unique name."""
    destination = f"{base_dir}{dataset_type}_{version}/images/"
    pairs = datasets[["orig_path", "filename"]].drop_duplicates()
    for orig_path, packaged_name in zip(pairs["orig_path"], pairs["filename"]):
        dst = os.path.join(destination, packaged_name)
        if not os.path.exists(dst):
            shutil.copy(orig_path, dst)


def copy_masks(datasets, base_dir, dataset_type, mask_source_dir):
    """Copy precomputed tree coverage masks (keyed by the source-unique name).

    Returns a filtered dataset with rows removed for any images missing a mask.
    """
    mask_source_dir = Path(mask_source_dir)
    destination = f"{base_dir}{dataset_type}_{version}/masks/"
    missing_images = set()
    for packaged_name in datasets["filename"].unique():
        mask_name = f"{Path(packaged_name).stem}.png"
        source_mask = mask_source_dir / mask_name
        if not source_mask.exists():
            print(f"Warning: Missing tree coverage mask for {packaged_name}, removing from dataset.")
            missing_images.add(packaged_name)
            continue
        destination_mask = os.path.join(destination, mask_name)
        if not os.path.exists(destination_mask):
            shutil.copy(source_mask, destination_mask)
    if missing_images:
        datasets = datasets[~datasets["filename"].isin(missing_images)].copy()
    return datasets


def copy_packaged_assets_from_full(base_dir, dataset_type, version, filenames,
                                   suffix, subdir):
    """Copy already-packaged assets (images / masks) into suffixed dataset folders."""
    source_dir = Path(f"{base_dir}{dataset_type}_{version}/{subdir}")
    dest_dir = Path(f"{base_dir}{dataset_type}{suffix}_{version}/{subdir}")
    os.makedirs(dest_dir, exist_ok=True)
    for filename in set(filenames):
        if subdir == "images":
            src_name = filename
        else:
            src_name = f"{Path(filename).stem}.png"
        src = source_dir / src_name
        dst = dest_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f"Missing packaged {subdir[:-1]} file: {src}")
        if not dst.exists():
            shutil.copy(src, dst)

from milliontrees.common.release_sizes import MINI_IMAGES_PER_SOURCE, SMALL_IMAGES_PER_SOURCE


def _top_filenames_per_source(datasets, n_per_source):
    """Return up to n_per_source filenames per source (by annotation count)."""
    filename_counts = (
        datasets.groupby(["source", "filename"]).size().reset_index(name="count"))
    top_per_source = (
        filename_counts.sort_values("count", ascending=False)
        .groupby("source", group_keys=False)
        .apply(lambda g: g.head(n_per_source))
        .reset_index(drop=True)
    )
    return top_per_source["filename"].unique().tolist()


def apply_existing_splits(df):
    """Honor pre-assigned train, test, or validation splits from existing_split."""
    df = df.copy()
    if "existing_split" not in df.columns:
        return df
    for split_name in ("train", "test", "validation"):
        mask = df["existing_split"] == split_name
        df.loc[mask, "split"] = split_name
    return df


def _validation_sources(df):
    """Return sources whose rows are pinned to the validation split."""
    if "existing_split" not in df.columns:
        return set()
    return set(
        df.loc[df["existing_split"] == "validation", "source"].dropna().unique())


def _rows_needing_auto_split(df):
    """Rows without a pre-assigned split from existing_split."""
    if "existing_split" not in df.columns:
        return pd.Series(True, index=df.index)
    pinned = df["existing_split"].isin(["train", "test", "validation"])
    return ~pinned


def _ensure_test_split(annotations):
    """Ensure at least one test image so val loaders are non-empty."""
    if "split" not in annotations.columns:
        return annotations
    if (annotations["split"] == "validation").any() and (annotations["split"] == "test").sum() == 0:
        return annotations
    if (annotations["split"] == "test").sum() == 0 and len(annotations) > 0:
        first_filename = annotations["filename"].iloc[0]
        annotations.loc[annotations["filename"] == first_filename, "split"] = "test"
    return annotations


def create_mini_datasets(datasets, base_dir, dataset_type, version):
    """Create mini datasets for debugging and generate visualizations.
    Takes the top MINI_IMAGES_PER_SOURCE images per source (by annotation count).
    """
    mini_filenames = _top_filenames_per_source(datasets, MINI_IMAGES_PER_SOURCE)
    mini_annotations = _ensure_test_split(
        datasets[datasets["filename"].isin(mini_filenames)].copy())
    mini_annotations.to_csv(f"{base_dir}Mini{dataset_type}_{version}/random.csv", index=False)
    
    # Copy images for mini datasets
    for image in mini_filenames:
        destination = f"{base_dir}Mini{dataset_type}_{version}/images/"
        shutil.copy(f"{base_dir}{dataset_type}_{version}/images/" + image, destination)
        mask_name = f"{Path(image).stem}.png"
        mini_mask_destination = f"{base_dir}Mini{dataset_type}_{version}/masks/"
        shutil.copy(f"{base_dir}{dataset_type}_{version}/masks/" + mask_name, mini_mask_destination)

    # Generate visualizations for each source (one image per source to avoid overlaying
    # annotations from multiple images onto a single background image)
    for source, group in mini_annotations.groupby("source"):
        print(source)
        # Use only the first image to avoid mixing coordinates from different images
        first_image = group["filename"].iloc[0]
        group = group[group["filename"] == first_image].copy()
        group["image_path"] = group["filename"]
        group = read_file(group, root_dir=f"{base_dir}Mini{dataset_type}_{version}/images/")
        group.root_dir = f"{base_dir}Mini{dataset_type}_{version}/images/"

        # Remove spaces in source name
        source = source.replace(" ", "_")

        # Handle polygons specifically to include image dimensions
        height, width, channels = cv2.imread(f"{base_dir}Mini{dataset_type}_{version}/images/" + group.image_path.iloc[0]).shape
        plot_results(group, savedir="docs/public/", basename=source)


def create_small_datasets(datasets, base_dir, dataset_type, version):
    """Create small release datasets (up to SMALL_IMAGES_PER_SOURCE images per source)."""
    small_filenames = _top_filenames_per_source(datasets, SMALL_IMAGES_PER_SOURCE)
    small_annotations = _ensure_test_split(
        datasets[datasets["filename"].isin(small_filenames)].copy())

    for image in small_filenames:
        destination = f"{base_dir}Small{dataset_type}_{version}/images/"
        shutil.copy(f"{base_dir}{dataset_type}_{version}/images/" + image, destination)
        mask_name = f"{Path(image).stem}.png"
        shutil.copy(
            f"{base_dir}{dataset_type}_{version}/masks/" + mask_name,
            f"{base_dir}Small{dataset_type}_{version}/masks/",
        )

    return small_annotations


def create_release_files(base_dir, dataset_type, version, prefix=""):
    """Create release files for the dataset."""
    with open(
            f"{base_dir}{prefix}{dataset_type}_{version}/RELEASE_{version}.txt",
            "w",
    ) as outfile:
        outfile.write(f"Version: {version}")

def process_splits_and_release(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version, suffix="", prefix=""):
    """Wrapper function to perform splits and create release files for datasets.
    
    Args:
        TreePolygons_datasets: DataFrame with polygon annotations
        TreePoints_datasets: DataFrame with point annotations  
        TreeBoxes_datasets: DataFrame with box annotations
        base_dir: Base directory for output
        version: Version string
        suffix: Optional suffix to append to directory names (e.g., "_supervised")
        prefix: Optional prefix for directory names (e.g., "Small" for small releases)
    """
    if suffix or prefix:
        adjusted_base_dir = base_dir
        for dataset_type in ["TreeBoxes", "TreePoints", "TreePolygons"]:
            os.makedirs(
                f"{adjusted_base_dir}{prefix}{dataset_type}{suffix}_{version}",
                exist_ok=True,
            )
            os.makedirs(
                f"{adjusted_base_dir}{prefix}{dataset_type}{suffix}_{version}/images",
                exist_ok=True,
            )
            os.makedirs(
                f"{adjusted_base_dir}{prefix}{dataset_type}{suffix}_{version}/masks",
                exist_ok=True,
            )
    
        
    # Select columns
    columns_to_keep = ["filename", "geometry", "source", "split", "complete",
                       "xmin", "ymin", "xmax", "ymax", "x", "y", "polygon","existing_split"]


    TreePolygons_datasets = keep_columns_if_exist(TreePolygons_datasets, columns_to_keep)
    TreePoints_datasets = keep_columns_if_exist(TreePoints_datasets, columns_to_keep)
    TreeBoxes_datasets = keep_columns_if_exist(TreeBoxes_datasets, columns_to_keep)

    zero_shot_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets,
                    base_dir, version, suffix, prefix)
    random_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets,
                 base_dir, version, suffix, prefix)
    cross_geometry_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets,
                         base_dir, version, suffix, prefix)

    create_release_files(base_dir, f"TreeBoxes{suffix}", version, prefix)
    create_release_files(base_dir, f"TreePoints{suffix}", version, prefix)
    create_release_files(base_dir, f"TreePolygons{suffix}", version, prefix)

def zip_directory(folder_path, zip_path):
    """Zip the contents of a directory."""
    # Remove the existing zip file if it exists
    if os.path.exists(zip_path):
        os.remove(zip_path)
    # Create a new zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def random_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets,
                 base_dir, version, suffix="", prefix=""):
    """Perform random split and save the results."""
    
    def apply_split(df):
        df = apply_existing_splits(df)
        needs_split = _rows_needing_auto_split(df)
        remaining = df.loc[needs_split].copy()
        if not remaining.empty:
            for source in remaining["source"].dropna().unique():
                idx = remaining[remaining["source"] == source].index
                n = len(idx)
                if n == 0:
                    continue
                n_test = max(1, int(round(n * 0.1))) if n > 1 else 0
                test_idx = idx[:n_test]
                train_idx = idx[n_test:]
                if n_test > 0:
                    df.loc[test_idx, "split"] = "test"
                if len(train_idx) > 0:
                    df.loc[train_idx, "split"] = "train"
        return df

    TreePolygons_datasets = apply_split(TreePolygons_datasets)
    TreePoints_datasets = apply_split(TreePoints_datasets)
    TreeBoxes_datasets = apply_split(TreeBoxes_datasets)

    # Remove from test split any entries with 'unsupervised' or 'weak supervised'
    # in the source column. normalize_unsupervised_sources() guarantees aliases
    # like SPREAD/Feng already carry the 'unsupervised' suffix, so a single
    # substring check covers them.
    def remove_unsupervised_test_entries(df):
        if "split" in df.columns and "source" in df.columns:
            mask = (df["split"] == "test") & (df["source"].str.contains("unsupervised|weak supervised", case=False, na=False))
            return df.loc[~mask]
        return df

    TreePolygons_datasets = remove_unsupervised_test_entries(TreePolygons_datasets)
    TreePoints_datasets = remove_unsupervised_test_entries(TreePoints_datasets)
    TreeBoxes_datasets = remove_unsupervised_test_entries(TreeBoxes_datasets)
    
    # Limit test images to 100 per source
    # Get all sources that have test data
    def get_test_sources(df):
        if "split" in df.columns and "source" in df.columns:
            test_data = df[df["split"] == "test"]
            return test_data["source"].unique().tolist() if not test_data.empty else []
        return []
    
    test_sources_polygons = get_test_sources(TreePolygons_datasets)
    test_sources_points = get_test_sources(TreePoints_datasets)
    test_sources_boxes = get_test_sources(TreeBoxes_datasets)
    
    TreePolygons_datasets = limit_test_images(TreePolygons_datasets, test_sources_polygons)
    TreePoints_datasets = limit_test_images(TreePoints_datasets, test_sources_points)
    TreeBoxes_datasets = limit_test_images(TreeBoxes_datasets, test_sources_boxes)
   
   # Save the splits to CSV
    TreePolygons_datasets.to_csv(
        f"{base_dir}{prefix}TreePolygons{suffix}_{version}/random.csv", index=False)
    TreePoints_datasets.to_csv(
        f"{base_dir}{prefix}TreePoints{suffix}_{version}/random.csv", index=False)
    TreeBoxes_datasets.to_csv(
        f"{base_dir}{prefix}TreeBoxes{suffix}_{version}/random.csv", index=False)

    label = f"{prefix}{suffix}".strip("_") or "full"
    print(f"random splits saved ({label}):")
    print(f"TreePolygons: {base_dir}{prefix}TreePolygons{suffix}_{version}/random.csv")
    print(f"TreePoints: {base_dir}{prefix}TreePoints{suffix}_{version}/random.csv")
    print(f"TreeBoxes: {base_dir}{prefix}TreeBoxes{suffix}_{version}/random.csv")

def cross_geometry_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets,
                         base_dir, version, suffix="", prefix=""):
    """Perform cross-geometry split and save the results."""
    TreePolygons_datasets = apply_existing_splits(TreePolygons_datasets)
    TreePoints_datasets = apply_existing_splits(TreePoints_datasets)
    TreeBoxes_datasets = apply_existing_splits(TreeBoxes_datasets)

    TreePolygons_datasets.loc[
        TreePolygons_datasets["split"] != "validation", "split"] = "test"
    TreePolygons_datasets = TreePolygons_datasets[
        TreePolygons_datasets.source.isin(ZEROSHOT_TEST_SOURCES_POLYGONS)
        | (TreePolygons_datasets["split"] == "validation")
    ]
    TreePoints_datasets.loc[
        TreePoints_datasets["split"] != "validation", "split"] = "train"
    TreeBoxes_datasets.loc[
        TreeBoxes_datasets["split"] != "validation", "split"] = "train"

    # remove any unsupervised / weak supervised source from test (aliases such as
    # SPREAD/Feng are already suffixed by normalize_unsupervised_sources)
    unsupervised_mask_polygons = (TreePolygons_datasets.source.str.contains('unsupervised|weak supervised', case=False, na=False)) & (TreePolygons_datasets.split == "test")
    TreePolygons_datasets = TreePolygons_datasets[~unsupervised_mask_polygons]

    unsupervised_mask_points = (TreePoints_datasets.source.str.contains('unsupervised|weak supervised', case=False, na=False)) & (TreePoints_datasets.split == "test")
    TreePoints_datasets = TreePoints_datasets[~unsupervised_mask_points]

    unsupervised_mask_boxes = (TreeBoxes_datasets.source.str.contains('unsupervised|weak supervised', case=False, na=False)) & (TreeBoxes_datasets.split == "test")
    TreeBoxes_datasets = TreeBoxes_datasets[~unsupervised_mask_boxes]

    TreePolygons_datasets.to_csv(
        f"{base_dir}{prefix}TreePolygons{suffix}_{version}/crossgeometry.csv", index=False)
    TreePoints_datasets.to_csv(
        f"{base_dir}{prefix}TreePoints{suffix}_{version}/crossgeometry.csv", index=False)
    TreeBoxes_datasets.to_csv(
        f"{base_dir}{prefix}TreeBoxes{suffix}_{version}/crossgeometry.csv", index=False)

    label = f"{prefix}{suffix}".strip("_") or "full"
    print(f"Cross-geometry splits saved ({label}):")
    print(f"TreePolygons: {base_dir}{prefix}TreePolygons{suffix}_{version}/crossgeometry.csv")
    print(f"TreePoints: {base_dir}{prefix}TreePoints{suffix}_{version}/crossgeometry.csv")
    print(f"TreeBoxes: {base_dir}{prefix}TreeBoxes{suffix}_{version}/crossgeometry.csv")


# Limit test datasets to max_images per source (unless existing_split column exists)
def limit_test_images(df, test_sources, max_images=50, excess_mode="train"):
    """Limit test split to max_images per source.

    Sources already pinned via an ``existing_split == "test"`` column are honored
    in full and the cap is skipped.

    excess_mode controls what happens to test images beyond the cap:
      - "train": demote excess to the train split. Use for random splits, where
        the held-out source already has rows in train by construction.
      - "drop":  drop excess rows entirely so they appear in neither split. Use
        for zero-shot splits, where held-out sources must never leak into train.
    """
    if excess_mode not in {"train", "drop"}:
        raise ValueError(f"excess_mode must be 'train' or 'drop', got {excess_mode!r}")

    if "existing_split" in df.columns and "test" in df["existing_split"].unique():
        return df

    df = df.copy()
    test_data = df[df["split"] == "test"]

    drop_indices = []
    for source in test_sources:
        source_test = test_data[test_data["source"] == source]
        if source_test.empty:
            continue

        unique_images = source_test["filename"].unique()
        if len(unique_images) <= max_images:
            continue

        selected_images = pd.Series(unique_images).sample(n=max_images, random_state=42)
        excess_mask = (
            (df["source"] == source)
            & (df["split"] == "test")
            & (~df["filename"].isin(selected_images))
        )

        if excess_mode == "train":
            df.loc[excess_mask, "split"] = "train"
        else:
            drop_indices.extend(df.index[excess_mask].tolist())

    if drop_indices:
        df = df.drop(index=drop_indices)

    return df
        
# Zero-shot split
def zero_shot_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets,
                    base_dir, version, suffix="", prefix=""):
    """Perform zero-shot split and save the results."""
    # Define test and train sources
    test_sources_polygons = ZEROSHOT_TEST_SOURCES_POLYGONS
    train_sources_polygons = [x for x in TreePolygons_datasets.source.unique() if x not in test_sources_polygons]

    test_sources_points = ZEROSHOT_TEST_SOURCES_POINTS
    train_sources_points = [x for x in TreePoints_datasets.source.unique() if x not in test_sources_points]

    test_sources_boxes = ZEROSHOT_TEST_SOURCES_BOXES

    val_polygons = _validation_sources(TreePolygons_datasets)
    val_points = _validation_sources(TreePoints_datasets)
    val_boxes = _validation_sources(TreeBoxes_datasets)

    train_sources_polygons = [
        x for x in TreePolygons_datasets.source.unique()
        if x not in test_sources_polygons and x not in val_polygons
    ]
    train_sources_points = [
        x for x in TreePoints_datasets.source.unique()
        if x not in test_sources_points and x not in val_points
    ]
    train_sources_boxes = [
        x for x in TreeBoxes_datasets.source.unique()
        if x not in test_sources_boxes and x not in val_boxes
    ]

    TreePolygons_datasets = apply_existing_splits(TreePolygons_datasets)
    TreePoints_datasets = apply_existing_splits(TreePoints_datasets)
    TreeBoxes_datasets = apply_existing_splits(TreeBoxes_datasets)

    TreePolygons_datasets.loc[
        TreePolygons_datasets.source.isin(train_sources_polygons)
        & _rows_needing_auto_split(TreePolygons_datasets),
        "split",
    ] = "train"
    TreePolygons_datasets.loc[
        TreePolygons_datasets.source.isin(test_sources_polygons)
        & _rows_needing_auto_split(TreePolygons_datasets),
        "split",
    ] = "test"

    TreePoints_datasets.loc[
        TreePoints_datasets.source.isin(train_sources_points)
        & _rows_needing_auto_split(TreePoints_datasets),
        "split",
    ] = "train"
    TreePoints_datasets.loc[
        TreePoints_datasets.source.isin(test_sources_points)
        & _rows_needing_auto_split(TreePoints_datasets),
        "split",
    ] = "test"

    TreeBoxes_datasets.loc[
        TreeBoxes_datasets.source.isin(train_sources_boxes)
        & _rows_needing_auto_split(TreeBoxes_datasets),
        "split",
    ] = "train"
    TreeBoxes_datasets.loc[
        TreeBoxes_datasets.source.isin(test_sources_boxes)
        & _rows_needing_auto_split(TreeBoxes_datasets),
        "split",
    ] = "test"

    # Zero-shot: drop excess images instead of demoting to train so held-out
    # sources never leak into the train split.
    TreePolygons_datasets = limit_test_images(
        TreePolygons_datasets, test_sources_polygons, excess_mode="drop"
    )
    TreePoints_datasets = limit_test_images(
        TreePoints_datasets, test_sources_points, excess_mode="drop"
    )
    TreeBoxes_datasets = limit_test_images(
        TreeBoxes_datasets, test_sources_boxes, excess_mode="drop"
    )

    # Move any unsupervised / weak supervised source out of test (shouldn't be
    # there, but filter for safety). Aliases like SPREAD/Feng are already
    # suffixed by normalize_unsupervised_sources, so one substring check covers them.
    def remove_feng_and_unsupervised_from_test(df):
        if "split" in df.columns and "source" in df.columns:
            mask = (df["split"] == "test") & (df["source"].str.contains("unsupervised|weak supervised", case=False, na=False))
            if mask.any():
                df.loc[mask, "split"] = "train"
        return df
    
    TreePolygons_datasets = remove_feng_and_unsupervised_from_test(TreePolygons_datasets)
    TreePoints_datasets = remove_feng_and_unsupervised_from_test(TreePoints_datasets)
    TreeBoxes_datasets = remove_feng_and_unsupervised_from_test(TreeBoxes_datasets)

    TreePolygons_datasets.to_csv(
        f"{base_dir}{prefix}TreePolygons{suffix}_{version}/zeroshot.csv", index=False)
    TreePoints_datasets.to_csv(
        f"{base_dir}{prefix}TreePoints{suffix}_{version}/zeroshot.csv", index=False)
    TreeBoxes_datasets.to_csv(
        f"{base_dir}{prefix}TreeBoxes{suffix}_{version}/zeroshot.csv", index=False)

    label = f"{prefix}{suffix}".strip("_") or "full"
    print(f"Zero-shot splits saved ({label}):")
    print(f"TreePolygons: {base_dir}{prefix}TreePolygons{suffix}_{version}/zeroshot.csv")
    print(f"TreePoints: {base_dir}{prefix}TreePoints{suffix}_{version}/zeroshot.csv")
    print(f"TreeBoxes: {base_dir}{prefix}TreeBoxes{suffix}_{version}/zeroshot.csv")

def filter_out_unsupervised(datasets):
    """Filter out datasets with 'unsupervised' or 'weak supervised' in the source name."""
    return datasets[~datasets['source'].str.contains('unsupervised|weak supervised', case=False, na=False)]

def check_for_updated_annotations(dataset, geometry):
    updated_annotations = [pd.read_csv(x) for x in glob.glob(f"data_prep/annotations/*{geometry}*.csv")]
    if not updated_annotations:
        return dataset
        
    updated_annotations = pd.concat(updated_annotations)
    dataset["basename"] = dataset["filename"].str.split('/').str[-1]

    # Remove images marked for removal (vectorized)
    images_to_remove = updated_annotations[updated_annotations.remove == "Remove image from benchmark"]["image_path"].unique()
    dataset = dataset[~dataset["basename"].isin(images_to_remove)]

    # Filter valid annotations
    updated_annotations = updated_annotations[updated_annotations["label"].notna()]
    
    # Use set intersection for fast lookup
    updated_filenames = set(updated_annotations["image_path"].unique())
    dataset_filenames = set(dataset["basename"].unique())
    matching_files = updated_filenames & dataset_filenames
    
    if not matching_files:
        return dataset
        
    # Create mappings from basename to root_dir and source for each file
    # This ensures we use the correct root_dir for each source
    # We need to capture this BEFORE removing old annotations
    basename_to_root_dir = {}
    basename_to_source = {}
    for _, row in dataset.iterrows():
        basename = row["basename"]
        if basename in matching_files:
            if basename not in basename_to_root_dir:
                basename_to_root_dir[basename] = os.path.dirname(row["filename"])
            if basename not in basename_to_source:
                basename_to_source[basename] = row["source"]
    
    # Remove all old annotations at once
    dataset = dataset[~dataset["basename"].isin(matching_files)]
    
    # Process all updates in batch
    new_annotations = []
    skipped_files = []
    for filename in matching_files:
        # Get the original root_dir and source for this specific file
        root_dir = basename_to_root_dir.get(filename)
        original_source = basename_to_source.get(filename, "Unknown")
        
        if root_dir is None:
            # Fallback: use first filename's directory (shouldn't happen, but safe)
            root_dir = os.path.dirname(dataset["filename"].iloc[0]) if len(dataset) > 0 else ""
        
        updated_batch = updated_annotations[updated_annotations["image_path"] == filename].copy()
        updated_batch["source"] = original_source
        updated_batch["filename"] = updated_batch["image_path"].apply(lambda x: os.path.join(root_dir, x))
        
        # Check if image file exists before processing
        image_path = updated_batch["filename"].iloc[0] if len(updated_batch) > 0 else None
        if image_path and not os.path.exists(image_path):
            print(f"Warning: Image file does not exist, skipping annotations for {filename}: {image_path}")
            skipped_files.append((filename, image_path))
            continue
        
        # Process with read_file using the correct root_dir for this file
        updated_batch = read_file(updated_batch, root_dir=root_dir)
        
        # Filter out rows with None geometries (e.g., from empty images or invalid annotations)
        if "geometry" in updated_batch.columns:
            none_count = updated_batch["geometry"].isna().sum()
            if none_count > 0:
                print(f"Warning: Found {none_count} annotations with None geometry for {filename}, filtering them out")
            updated_batch = updated_batch[updated_batch["geometry"].notna()].copy()
        
        if len(updated_batch) > 0:
            new_annotations.append(updated_batch)
        else:
            print(f"Warning: No valid annotations after processing for {filename}")
            skipped_files.append((filename, "No valid geometries"))
    
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files with issues during annotation update")
    
    if new_annotations:
        # Single concat operation
        all_new = pd.concat(new_annotations, ignore_index=True)
        dataset = pd.concat([dataset, all_new], ignore_index=True)
    
    return dataset
def load_annotation_csvs(cfg_path=None):
    """Parse data_prep/annotation_csvs.cfg and return (TreeBoxes, TreePoints, TreePolygons) lists."""
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "annotation_csvs.cfg")
    sections = {"TreeBoxes": [], "TreePoints": [], "TreePolygons": []}
    current = None
    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
            elif current in sections:
                sections[current].append(line)
    return sections["TreeBoxes"], sections["TreePoints"], sections["TreePolygons"]


def run(version, base_dir, mask_source_dir=None, debug=False):
    if mask_source_dir is None:
        mask_source_dir = os.environ.get("MILLIONTREES_MASKS_DIR")
    if mask_source_dir is None:
        raise ValueError(
            "mask_source_dir is required. Pass it directly or set MILLIONTREES_MASKS_DIR."
        )
    TreeBoxes, TreePoints, TreePolygons = load_annotation_csvs()

    # Build the source-unique packaged-name map from the same CSV union the mask
    # precompute uses, so image/mask filenames agree between the two scripts.
    packaged_name_map = build_unique_name_map(
        collect_image_source_pairs(TreeBoxes + TreePoints + TreePolygons)
    )

    # Combine datasets
    TreeBoxes_datasets = combine_datasets(TreeBoxes, debug=debug)
    TreePoints_datasets = combine_datasets(TreePoints, debug=debug)
    TreePolygons_datasets = combine_datasets(TreePolygons, debug=debug)

    # Enforce canonical unsupervised source names up front so the label is baked
    # into every released CSV (and caught by the loader's '*unsupervised*'
    # exclusion) regardless of whether upstream generators were re-run.
    TreeBoxes_datasets = normalize_unsupervised_sources(TreeBoxes_datasets)
    TreePoints_datasets = normalize_unsupervised_sources(TreePoints_datasets)
    TreePolygons_datasets = normalize_unsupervised_sources(TreePolygons_datasets)

    # Fail loudly if any expected unsupervised source lost its suffix upstream.
    verify_unsupervised_sources(TreeBoxes_datasets, "TreeBoxes")
    verify_unsupervised_sources(TreePoints_datasets, "TreePoints")
    verify_unsupervised_sources(TreePolygons_datasets, "TreePolygons")

    # Coerce box columns early (avoids mixed str/float rows) and drop degenerate rows
    for col in ("xmin", "ymin", "xmax", "ymax"):
        TreeBoxes_datasets[col] = pd.to_numeric(TreeBoxes_datasets[col], errors="coerce")
    TreeBoxes_datasets = TreeBoxes_datasets.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    TreeBoxes_datasets = TreeBoxes_datasets[
        (TreeBoxes_datasets["xmax"] > TreeBoxes_datasets["xmin"])
        & (TreeBoxes_datasets["ymax"] > TreeBoxes_datasets["ymin"])
    ]

    # Check for updated annotations (from Label Studio review) and apply them
    TreeBoxes_datasets = check_for_updated_annotations(TreeBoxes_datasets, "Boxes")
    TreePoints_datasets = check_for_updated_annotations(TreePoints_datasets, "Points")
    # TreePolygons_datasets = check_for_updated_annotations(TreePolygons_datasets, "Polygons")

    # Split datasets
    TreeBoxes_datasets = split_dataset(TreeBoxes_datasets)
    TreePoints_datasets = split_dataset(TreePoints_datasets)
    TreePolygons_datasets = split_dataset(TreePolygons_datasets)

    # Create directories
    create_directories(base_dir, "TreeBoxes")
    create_directories(base_dir, "TreePoints")
    create_directories(base_dir, "TreePolygons")

    # Process geometry columns
    TreeBoxes_datasets = process_geometry_columns(TreeBoxes_datasets, "box")
    TreePoints_datasets = process_geometry_columns(TreePoints_datasets, "point")
    TreePolygons_datasets = process_geometry_columns(TreePolygons_datasets, "polygon")

    # Remove degenerate boxes (zero width/height) so albumentations does not raise
    TreeBoxes_datasets = filter_degenerate_boxes(TreeBoxes_datasets)

    # Remove degenerate polygons (those that would create invalid bounding boxes)
    TreePolygons_datasets = filter_degenerate_polygons(TreePolygons_datasets)

    # Defensive guard: collapse exact duplicate annotations (same image + same
    # geometry within a source). Upstream prediction CSVs have shipped doubled
    # boxes before (e.g. NEON unsupervised), which silently inflated release
    # counts. Geometry is now normalized to WKT, so identity is well-defined.
    TreeBoxes_datasets = drop_duplicate_annotations(TreeBoxes_datasets, "TreeBoxes")
    TreePoints_datasets = drop_duplicate_annotations(TreePoints_datasets, "TreePoints")
    TreePolygons_datasets = drop_duplicate_annotations(TreePolygons_datasets, "TreePolygons")

    # Assign source-unique packaged filenames (<stem>_<source><ext>) so basenames
    # don't collide across sources; keeps the original path in 'orig_path' for copy.
    TreeBoxes_datasets = assign_packaged_filenames(TreeBoxes_datasets, packaged_name_map)
    TreePoints_datasets = assign_packaged_filenames(TreePoints_datasets, packaged_name_map)
    TreePolygons_datasets = assign_packaged_filenames(TreePolygons_datasets, packaged_name_map)

    # Copy images
    copy_images(TreeBoxes_datasets, base_dir, "TreeBoxes")
    copy_images(TreePoints_datasets, base_dir, "TreePoints")
    copy_images(TreePolygons_datasets, base_dir, "TreePolygons")
    TreeBoxes_datasets = copy_masks(TreeBoxes_datasets, base_dir, "TreeBoxes", mask_source_dir)
    TreePoints_datasets = copy_masks(TreePoints_datasets, base_dir, "TreePoints", mask_source_dir)
    TreePolygons_datasets = copy_masks(TreePolygons_datasets, base_dir, "TreePolygons", mask_source_dir)

    # 'filename' now holds the packaged basename; drop the source-path helper column
    # so absolute server paths don't leak into the published CSVs.
    TreeBoxes_datasets = TreeBoxes_datasets.drop(columns=["orig_path"])
    TreePoints_datasets = TreePoints_datasets.drop(columns=["orig_path"])
    TreePolygons_datasets = TreePolygons_datasets.drop(columns=["orig_path"])

    create_mini_datasets(TreeBoxes_datasets, base_dir, "TreeBoxes", version)
    create_mini_datasets(TreePoints_datasets, base_dir, "TreePoints", version)
    create_mini_datasets(TreePolygons_datasets, base_dir, "TreePolygons", version)

    print("\n=== Processing SMALL release (up to 50 images per source) ===")
    TreeBoxes_small = create_small_datasets(TreeBoxes_datasets, base_dir, "TreeBoxes", version)
    TreePoints_small = create_small_datasets(TreePoints_datasets, base_dir, "TreePoints", version)
    TreePolygons_small = create_small_datasets(
        TreePolygons_datasets, base_dir, "TreePolygons", version)
    process_splits_and_release(
        TreePolygons_small.copy(),
        TreePoints_small.copy(),
        TreeBoxes_small.copy(),
        base_dir,
        version,
        prefix="Small",
    )

    # Process all sources (including unsupervised)
    print("\n=== Processing ALL sources (including unsupervised) ===")
    process_splits_and_release(
        TreePolygons_datasets.copy(), 
        TreePoints_datasets.copy(), 
        TreeBoxes_datasets.copy(), 
        base_dir, 
        version
    )
    
    # Process supervised sources only (excluding unsupervised)
    print("\n=== Processing SUPERVISED sources only (excluding unsupervised) ===")
    TreeBoxes_supervised = filter_out_unsupervised(TreeBoxes_datasets)
    TreePoints_supervised = filter_out_unsupervised(TreePoints_datasets)
    TreePolygons_supervised = filter_out_unsupervised(TreePolygons_datasets)
    
    print(f"Filtered datasets:")
    print(f"  TreeBoxes: {len(TreeBoxes_datasets)} -> {len(TreeBoxes_supervised)} samples")
    print(f"  TreePoints: {len(TreePoints_datasets)} -> {len(TreePoints_supervised)} samples") 
    print(f"  TreePolygons: {len(TreePolygons_datasets)} -> {len(TreePolygons_supervised)} samples")
    
    process_splits_and_release(
        TreePolygons_supervised,
        TreePoints_supervised, 
        TreeBoxes_supervised,
        base_dir,
        version,
        suffix="_supervised"
    )
    copy_packaged_assets_from_full(
        base_dir, "TreeBoxes", version, TreeBoxes_supervised["filename"], "_supervised", "images"
    )
    copy_packaged_assets_from_full(
        base_dir, "TreePoints", version, TreePoints_supervised["filename"], "_supervised", "images"
    )
    copy_packaged_assets_from_full(
        base_dir, "TreePolygons", version, TreePolygons_supervised["filename"], "_supervised", "images"
    )
    copy_packaged_assets_from_full(
        base_dir, "TreeBoxes", version, TreeBoxes_supervised["filename"], "_supervised", "masks"
    )
    copy_packaged_assets_from_full(
        base_dir, "TreePoints", version, TreePoints_supervised["filename"], "_supervised", "masks"
    )
    copy_packaged_assets_from_full(
        base_dir, "TreePolygons", version, TreePolygons_supervised["filename"], "_supervised", "masks"
    )

    # Zip datasets (commented out for large datasets to save space/time)
    zip_directory(f"{base_dir}TreeBoxes_{version}", f"{base_dir}TreeBoxes_{version}.zip")
    zip_directory(f"{base_dir}TreePoints_{version}", f"{base_dir}TreePoints_{version}.zip") 
    zip_directory(f"{base_dir}TreePolygons_{version}", f"{base_dir}TreePolygons_{version}.zip")
    
    # Zip supervised datasets
    zip_directory(f"{base_dir}TreeBoxes_supervised_{version}", f"{base_dir}TreeBoxes_supervised_{version}.zip")
    zip_directory(f"{base_dir}TreePoints_supervised_{version}", f"{base_dir}TreePoints_supervised_{version}.zip")
    zip_directory(f"{base_dir}TreePolygons_supervised_{version}", f"{base_dir}TreePolygons_supervised_{version}.zip")
    
    zip_directory(f"{base_dir}MiniTreeBoxes_{version}", f"{base_dir}MiniTreeBoxes_{version}.zip")
    zip_directory(f"{base_dir}MiniTreePoints_{version}", f"{base_dir}MiniTreePoints_{version}.zip")
    zip_directory(f"{base_dir}MiniTreePolygons_{version}", f"{base_dir}MiniTreePolygons_{version}.zip")
    zip_directory(f"{base_dir}SmallTreeBoxes_{version}", f"{base_dir}SmallTreeBoxes_{version}.zip")
    zip_directory(f"{base_dir}SmallTreePoints_{version}", f"{base_dir}SmallTreePoints_{version}.zip")
    zip_directory(f"{base_dir}SmallTreePolygons_{version}", f"{base_dir}SmallTreePolygons_{version}.zip")


if __name__ == "__main__":
    version = "v0.17"
    base_dir = "/orange/ewhite/web/public/MillionTrees/"
    mask_source_dir = "/orange/ewhite/DeepForest/tree_coverage_masks"
    debug = False
    run(version, base_dir, mask_source_dir, debug)