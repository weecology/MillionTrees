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


def combine_datasets(dataset_paths, debug=False):
    datasets = []
    for dataset_path in dataset_paths:
        df = pd.read_csv(dataset_path)
        if "image_path" in df.columns:
            if "filename" in df.columns:
                df = df.drop(columns="filename")  # Remove existing filename if present
            df = df.rename(columns={"image_path": "filename"})
            df.reset_index(drop=True, inplace=True)
        datasets.append(df)
    
    combined_df = pd.concat(datasets, ignore_index=True)
    combined_df["complete"] = True

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
    if geom_type == "box":
        datasets[["xmin", "ymin", "xmax", "ymax"]] = gpd.GeoSeries.from_wkt(datasets["geometry"]).bounds
    elif geom_type == "point":
        datasets["x"] = gpd.GeoSeries.from_wkt(datasets["geometry"]).centroid.x
        datasets["y"] = gpd.GeoSeries.from_wkt(datasets["geometry"]).centroid.y
    elif geom_type == "polygon":
        datasets["polygon"] = gpd.GeoDataFrame(datasets.geometry).to_wkt()
        # Remove multipolygons
        datasets = datasets[datasets["geometry"].apply(lambda x: gpd.GeoSeries.from_wkt([x]).geom_type[0] != "MultiPolygon")]
    return datasets


def create_directories(base_dir, dataset_type):
    """Create directories for the dataset."""
    os.makedirs(f"{base_dir}{dataset_type}_{version}/images", exist_ok=True)
    os.makedirs(f"{base_dir}Mini{dataset_type}_{version}/images", exist_ok=True)


def copy_images(datasets, base_dir, dataset_type):
    """Copy images to the destination folder."""
    for image in datasets["filename"].unique():
        destination = f"{base_dir}{dataset_type}_{version}/images/"
        if not os.path.exists(os.path.join(destination, os.path.basename(image))):
            shutil.copy(image, destination)

def create_mini_datasets(datasets, base_dir, dataset_type, version):
    """Create mini datasets for debugging and generate visualizations."""
    # For each source, get the filename with the most annotations
    filename_counts = datasets.groupby(["source", "filename"]).size().reset_index(name="count")
    max_count_indices = filename_counts.groupby("source")["count"].idxmax()
    max_count_filenames = filename_counts.loc[max_count_indices]
    
    # Get one row per source (the first row for each filename with max annotations)
    mini_datasets = []
    for _, row in max_count_filenames.iterrows():
        source_data = datasets[(datasets["source"] == row["source"]) & (datasets["filename"] == row["filename"])]
        mini_datasets.append(source_data.iloc[0])  # Take the first row
    
    mini_datasets = pd.DataFrame(mini_datasets)
    mini_filenames = mini_datasets["filename"].tolist()
    mini_annotations = datasets[datasets["filename"].isin(mini_filenames)]
    mini_annotations.to_csv(f"{base_dir}Mini{dataset_type}_{version}/random.csv", index=False)
    
    # Copy images for mini datasets
    for image in mini_filenames:
        destination = f"{base_dir}Mini{dataset_type}_{version}/images/"
        shutil.copy(f"{base_dir}{dataset_type}_{version}/images/" + image, destination)

    # Generate visualizations for each source
    for source, group in mini_annotations.groupby("source"):
        print(source)
        group["image_path"] = group["filename"]
        group = read_file(group, root_dir=f"{base_dir}Mini{dataset_type}_{version}/images/")
        group.root_dir = f"{base_dir}Mini{dataset_type}_{version}/images/"
        
        # Remove spaces in source name
        source = source.replace(" ", "_")
        
        # Handle polygons specifically to include image dimensions
        height, width, channels = cv2.imread(f"{base_dir}Mini{dataset_type}_{version}/images/" + group.image_path.iloc[0]).shape
        plot_results(group, savedir="docs/public/", basename=source, height=height, width=width)

def create_release_files(base_dir, dataset_type, version):
    """Create release files for the dataset."""
    with open(f"{base_dir}{dataset_type}_{version}/RELEASE_{version}.txt", "w") as outfile:
        outfile.write(f"Version: {version}")

def process_splits_and_release(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version, suffix=""):
    """Wrapper function to perform splits and create release files for datasets.
    
    Args:
        TreePolygons_datasets: DataFrame with polygon annotations
        TreePoints_datasets: DataFrame with point annotations  
        TreeBoxes_datasets: DataFrame with box annotations
        base_dir: Base directory for output
        version: Version string
        suffix: Optional suffix to append to directory names (e.g., "_supervised")
    """
    # Adjust base directory if suffix provided
    if suffix:
        adjusted_base_dir = base_dir
        # Update dataset directories with suffix
        for dataset_type in ["TreeBoxes", "TreePoints", "TreePolygons"]:
            os.makedirs(f"{adjusted_base_dir}{dataset_type}{suffix}_{version}", exist_ok=True)
    
        
    # Select columns
    columns_to_keep = ["filename", "geometry", "source", "split", "complete",
                       "xmin", "ymin", "xmax", "ymax", "x", "y", "polygon"]


    TreePolygons_datasets = keep_columns_if_exist(TreePolygons_datasets, columns_to_keep)
    TreePoints_datasets = keep_columns_if_exist(TreePoints_datasets, columns_to_keep)
    TreeBoxes_datasets = keep_columns_if_exist(TreeBoxes_datasets, columns_to_keep)

    # Perform splits
    zero_shot_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version, suffix)
    random_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version, suffix)
    cross_geometry_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version, suffix)

    # Create release files
    create_release_files(base_dir, f"TreeBoxes{suffix}", version)
    create_release_files(base_dir, f"TreePoints{suffix}", version)  
    create_release_files(base_dir, f"TreePolygons{suffix}", version)

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

def random_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version, suffix=""):
    """Perform random split and save the results."""
    
    # Helper function to create the "split" column based on instructions
    def apply_split(df):
        df = df.copy()
        # If "existing_split" is present and value is "test", set "split" to "test"
        if "existing_split" in df.columns:
            mask = df["existing_split"] == "test"
            df.loc[mask, "split"] = "test"
            needs_split = ~mask
        else:
            needs_split = pd.Series([True]*len(df), index=df.index)
        # For each source not yet assigned "test", random split 0.9/0.1
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

    # Remove from test split any entries with 'unsupervised' in the source column
    def remove_unsupervised_test_entries(df):
        if "split" in df.columns and "source" in df.columns:
            mask = (df["split"] == "test") & (df["source"].str.contains("unsupervised", case=False, na=False))
            return df.loc[~mask]
        return df

    TreePolygons_datasets = remove_unsupervised_test_entries(TreePolygons_datasets)
    TreePoints_datasets = remove_unsupervised_test_entries(TreePoints_datasets)
    TreeBoxes_datasets = remove_unsupervised_test_entries(TreeBoxes_datasets)
    # Save the splits to CSV
    TreePolygons_datasets.to_csv(f"{base_dir}TreePolygons{suffix}_{version}/random.csv", index=False)
    TreePoints_datasets.to_csv(f"{base_dir}TreePoints{suffix}_{version}/random.csv", index=False)
    TreeBoxes_datasets.to_csv(f"{base_dir}TreeBoxes{suffix}_{version}/random.csv", index=False)

    print(f"random splits saved{' ('+suffix.strip('_')+')' if suffix else ''}:")
    print(f"TreePolygons: {base_dir}TreePolygons{suffix}_{version}/random.csv")
    print(f"TreePoints: {base_dir}TreePoints{suffix}_{version}/random.csv")
    print(f"TreeBoxes: {base_dir}TreeBoxes{suffix}_{version}/random.csv")

def cross_geometry_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version, suffix=""):
    """Perform cross-geometry split and save the results."""
    # Assign all polygons to train, points to test, and boxes to test
    TreePolygons_datasets["split"] = "test"

    # Filter which datasets go to polygon tests
    TreePolygons_datasets = TreePolygons_datasets[TreePolygons_datasets.source.isin([
        "Troles et al. 2024",
        "Lefebvre et al. 2024"
    ])]
    TreePoints_datasets["split"] = "train"
    TreeBoxes_datasets["split"] = "train"

    # remove any source with unsupervised from test
    TreePolygons_datasets = TreePolygons_datasets[~((TreePolygons_datasets.source.str.contains('unsupervised', case=False, na=False)) & (TreePolygons_datasets.split == "test"))]
    TreePoints_datasets = TreePoints_datasets[~((TreePoints_datasets.source.str.contains('unsupervised', case=False, na=False)) & (TreePoints_datasets.split == "test"))]
    TreeBoxes_datasets = TreeBoxes_datasets[~((TreeBoxes_datasets.source.str.contains('unsupervised', case=False, na=False)) & (TreeBoxes_datasets.split == "test"))]

    # Save the splits to CSV (use consistent folder naming: <DatasetName><suffix>_<version>)
    TreePolygons_datasets.to_csv(f"{base_dir}TreePolygons{suffix}_{version}/crossgeometry.csv", index=False)
    TreePoints_datasets.to_csv(f"{base_dir}TreePoints{suffix}_{version}/crossgeometry.csv", index=False)
    TreeBoxes_datasets.to_csv(f"{base_dir}TreeBoxes{suffix}_{version}/crossgeometry.csv", index=False)

    print(f"Cross-geometry splits saved{' ('+suffix.strip('_')+')' if suffix else ''}:")
    print(f"TreePolygons: {base_dir}TreePolygons{suffix}_{version}/crossgeometry.csv")
    print(f"TreePoints: {base_dir}TreePoints{suffix}_{version}/crossgeometry.csv")
    print(f"TreeBoxes: {base_dir}TreeBoxes{suffix}_{version}/crossgeometry.csv")

# Zero-shot split
def zero_shot_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version, suffix=""):
    """Perform zero-shot split and save the results."""
    # Define test and train sources
    test_sources_polygons = ["Troles et al. 2024","Bolhman 2008"]
    train_sources_polygons = [x for x in TreePolygons_datasets.source.unique() if x not in test_sources_polygons]

    test_sources_points = ["Amirkolaee et al. 2023","NEON_points"]
    train_sources_points = [x for x in TreePoints_datasets.source.unique() if x not in test_sources_points]

    test_sources_boxes = ["Radogoshi et al. 2021","SelvaBox","NEON_benchmark"]
    train_sources_boxes = [x for x in TreeBoxes_datasets.source.unique() if x not in test_sources_boxes]

    # Assign splits for polygons
    TreePolygons_datasets.loc[TreePolygons_datasets.source.isin(train_sources_polygons), "split"] = "train"
    TreePolygons_datasets.loc[TreePolygons_datasets.source.isin(test_sources_polygons), "split"] = "test"

    # Assign splits for points
    TreePoints_datasets.loc[TreePoints_datasets.source.isin(train_sources_points), "split"] = "train"
    TreePoints_datasets.loc[TreePoints_datasets.source.isin(test_sources_points), "split"] = "test"

    # Assign splits for boxes
    TreeBoxes_datasets.loc[TreeBoxes_datasets.source.isin(train_sources_boxes), "split"] = "train"
    TreeBoxes_datasets.loc[TreeBoxes_datasets.source.isin(test_sources_boxes), "split"] = "test"

    # Save the splits to CSV
    TreePolygons_datasets.to_csv(f"{base_dir}TreePolygons{suffix}_{version}/zeroshot.csv", index=False)
    TreePoints_datasets.to_csv(f"{base_dir}TreePoints{suffix}_{version}/zeroshot.csv", index=False)
    TreeBoxes_datasets.to_csv(f"{base_dir}TreeBoxes{suffix}_{version}/zeroshot.csv", index=False)

    print(f"Zero-shot splits saved{' ('+suffix.strip('_')+')' if suffix else ''}:")
    print(f"TreePolygons: {base_dir}TreePolygons{suffix}_{version}/zeroshot.csv")
    print(f"TreePoints: {base_dir}TreePoints{suffix}_{version}/zeroshot.csv")
    print(f"TreeBoxes: {base_dir}TreeBoxes{suffix}_{version}/zeroshot.csv")

def filter_out_unsupervised(datasets):
    """Filter out datasets that contain 'unsupervised' in the source name."""
    return datasets[~datasets['source'].str.contains('unsupervised', case=False, na=False)]

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
        
    # Batch process all updates
    root_dir = os.path.dirname(dataset["filename"].iloc[0])
    
    # Remove all old annotations at once
    dataset = dataset[~dataset["basename"].isin(matching_files)]
    
    # Process all updates in batch
    new_annotations = []
    for filename in matching_files:
        original_source = dataset[dataset["basename"] == filename]["source"].iloc[0] if len(dataset[dataset["basename"] == filename]) > 0 else "Unknown"
        
        updated_batch = updated_annotations[updated_annotations["image_path"] == filename].copy()
        updated_batch["source"] = original_source
        updated_batch["filename"] = updated_batch["image_path"].apply(lambda x: os.path.join(root_dir, x))
        
        new_annotations.append(updated_batch)
    
    if new_annotations:
        # Single concat operation
        all_new = pd.concat(new_annotations, ignore_index=True)
        all_new = read_file(all_new, root_dir=root_dir)
        dataset = pd.concat([dataset, all_new], ignore_index=True)
    
    return dataset
def run(version, base_dir, debug=False):
    TreeBoxes = [
        #"/orange/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv",
        #"/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/annotations.csv",
        #'/orange/ewhite/DeepForest/individual_urban_tree_crown_detection/annotations.csv',
        '/orange/ewhite/DeepForest/Radogoshi_Sweden/annotations.csv',
        #"/orange/ewhite/DeepForest/WRI/WRI-labels-opensource/annotations.csv",
        "/orange/ewhite/DeepForest/Guangzhou2022/annotations.csv",
        "/orange/ewhite/DeepForest/NEON_benchmark/NeonTreeEvaluation_annotations.csv",
        "/orange/ewhite/DeepForest/NEON_benchmark/University_of_Florida.csv",
        '/orange/ewhite/DeepForest/ReForestTree/images/train.csv',
        #"/orange/ewhite/DeepForest/Santos2019/annotations.csv",
        "/orange/ewhite/DeepForest/Zenodo_15155081/parsed_annotations.csv",
        "/orange/ewhite/DeepForest/OAM_TCD/annotations.csv"
        ,"/orange/ewhite/DeepForest/Zenodo_15155081/parsed_annotations.csv",
        "/orange/ewhite/DeepForest/SelvaBox/annotations.csv",
        "/orange/ewhite/DeepForest/neon_unsupervised/TreeBoxes_neon_unsupervised.csv",
        "/orange/ewhite/DeepForest/OpenForestObservatory/images/TreeBoxes_OFO_unsupervised.csv"
        #"/orange/ewhite/DeepForest/Beloiu_2023/annotations.csv",
   ]

    TreePoints = [
        "/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv",
        "/orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv",
        "/orange/ewhite/MillionTrees/NEON_points/annotations.csv",
        #'/orange/ewhite/DeepForest/BohlmanBCI/crops/annotations_points.csv',
        "/orange/ewhite/DeepForest/AutoArborist/downloaded_imagery/AutoArborist_combined_annotations.csv",
        "/orange/ewhite/DeepForest/Yosemite/tiles/yosemite_all_annotations.csv",
        "/orange/ewhite/DeepForest/OpenForestObservatory/images/TreePoints_OFO_unsupervised.csv",
        #'/orange/ewhite/DeepForest/Miraki/annotations.csv'
    ]

    TreePolygons = [
        "/orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv",
        "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv",
        "/orange/ewhite/DeepForest/Cloutier2023/images/annotations.csv",
        "/orange/ewhite/DeepForest/Firoze2023/annotations.csv",
        #"/orange/ewhite/DeepForest/Wagner_Australia/annotations.csv",
        #"/orange/ewhite/DeepForest/Alejandro_Chile/alejandro/annotations.csv",
        "/orange/ewhite/DeepForest/UrbanLondon/annotations.csv",
        #"/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/annotations.csv",
        #"/orange/ewhite/DeepForest/Araujo_2020/annotations.csv",
        #"/orange/ewhite/DeepForest/justdiggit-drone/label_sample/annotations.csv",
        "/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/annotations.csv",
        "/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/annotations.csv",
        "/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/annotations.csv",
        "/orange/ewhite/DeepForest/SPREAD/annotations.csv",
        "/orange/ewhite/DeepForest/KagglePalm/Palm-Counting-349images/annotations.csv",
        "/orange/ewhite/DeepForest/Kattenborn/uav_newzealand_waititu/crops/annotations.csv",
        "/orange/ewhite/DeepForest/Quebec_Lefebvre/Dataset/Crops/annotations.csv",
        #"/orange/ewhite/DeepForest/BohlmanBCI/crops/annotations_crowns.csv",
        "/orange/ewhite/DeepForest/TreeCountSegHeight/extracted_data_2aux_v4_cleaned_centroid_raw 2/annotations.csv",
        "/orange/ewhite/DeepForest/Schutte_Germany/annotations.csv",
        #"/orange/ewhite/DeepForest/takeshige2025/crops/annotations.csv",
    ]
    
    # Combine datasets
    TreeBoxes_datasets = combine_datasets(TreeBoxes, debug=debug)
    TreePoints_datasets = combine_datasets(TreePoints, debug=debug)
    TreePolygons_datasets = combine_datasets(TreePolygons, debug=debug)

    # Remove rows where xmin equals xmax
    TreeBoxes_datasets = TreeBoxes_datasets[TreeBoxes_datasets["xmin"] != TreeBoxes_datasets["xmax"]]
    TreeBoxes_datasets = TreeBoxes_datasets[TreeBoxes_datasets["ymin"] != TreeBoxes_datasets["ymax"]]

    # Check for updated annotations
    check_for_updated_annotations(TreeBoxes_datasets, "Boxes")
    check_for_updated_annotations(TreePoints_datasets, "Points")
    #check_for_updated_annotations(TreePolygons_datasets, "Polygons")

    # Split datasets
    TreeBoxes_datasets = split_dataset(TreeBoxes_datasets)
    TreePoints_datasets = split_dataset(TreePoints_datasets)
    TreePolygons_datasets = split_dataset(TreePolygons_datasets)

    # Create directories
    create_directories(base_dir, "TreeBoxes")
    create_directories(base_dir, "TreePoints")
    create_directories(base_dir, "TreePolygons")

    # Save the default random split
    TreeBoxes_datasets.to_csv(f"{base_dir}TreeBoxes_{version}/random.csv", index=False)
    TreePoints_datasets.to_csv(f"{base_dir}TreePoints_{version}/random.csv", index=False)
    TreePolygons_datasets.to_csv(f"{base_dir}TreePolygons_{version}/random.csv", index=False)

    # Process geometry columns
    TreeBoxes_datasets = process_geometry_columns(TreeBoxes_datasets, "box")
    TreePoints_datasets = process_geometry_columns(TreePoints_datasets, "point")
    TreePolygons_datasets = process_geometry_columns(TreePolygons_datasets, "polygon")

    # Copy images
    copy_images(TreeBoxes_datasets, base_dir, "TreeBoxes")
    copy_images(TreePoints_datasets, base_dir, "TreePoints")
    copy_images(TreePolygons_datasets, base_dir, "TreePolygons")

    # change filenames to relative path
    TreeBoxes_datasets["filename"] = TreeBoxes_datasets["filename"].apply(os.path.basename)
    TreePoints_datasets["filename"] = TreePoints_datasets["filename"].apply(os.path.basename)
    TreePolygons_datasets["filename"] = TreePolygons_datasets["filename"].apply(os.path.basename)

    # Create mini datasets
    create_mini_datasets(TreeBoxes_datasets, base_dir, "TreeBoxes", version)
    create_mini_datasets(TreePoints_datasets, base_dir, "TreePoints", version)
    create_mini_datasets(TreePolygons_datasets, base_dir, "TreePolygons", version)

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

    # Zip datasets (commented out for large datasets to save space/time)
    zip_directory(f"{base_dir}TreeBoxes_{version}", f"{base_dir}TreeBoxes_{version}.zip")
    zip_directory(f"{base_dir}TreePoints_{version}", f"{base_dir}TreePoints_{version}.zip") 
    zip_directory(f"{base_dir}TreePolygons_{version}", f"{base_dir}TreePolygons_{version}.zip")
    
    # Zip supervised datasets
    zip_directory(f"{base_dir}TreeBoxes_supervised_{version}", f"{base_dir}TreeBoxes_supervised_{version}.zip")
    zip_directory(f"{base_dir}TreePoints_supervised_{version}", f"{base_dir}TreePoints_supervised_{version}.zip")
    zip_directory(f"{base_dir}TreePolygons_supervised_{version}", f"{base_dir}TreePolygons_supervised_{version}.zip")
    
    # Zip only mini datasets
    zip_directory(f"{base_dir}MiniTreeBoxes_{version}", f"{base_dir}MiniTreeBoxes_{version}.zip")
    zip_directory(f"{base_dir}MiniTreePoints_{version}", f"{base_dir}MiniTreePoints_{version}.zip")
    zip_directory(f"{base_dir}MiniTreePolygons_{version}", f"{base_dir}MiniTreePolygons_{version}.zip")


if __name__ == "__main__":
    version = "v0.8"
    base_dir = "/orange/ewhite/web/public/MillionTrees/"
    debug = False
    run(version, base_dir, debug)