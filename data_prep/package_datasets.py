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
    """Combine multiple datasets into a single DataFrame."""
    datasets = []
    for dataset in dataset_paths:
        datasets.append(pd.read_csv(dataset))
    
    df = pd.concat(datasets)
    df = df.rename(columns={"image_path":"filename"})
    if debug:
        df = df.groupby("source").head()

    return df


def split_dataset(datasets, split_column="filename", frac=0.8):
    """Split the dataset into training and testing sets."""
    train_images = datasets[split_column].drop_duplicates().sample(frac=frac)
    datasets.loc[datasets[split_column].isin(train_images), "split"] = "train"
    datasets.loc[~datasets[split_column].isin(train_images), "split"] = "test"
    return datasets


def process_geometry_columns(datasets, geom_type):
    """Process geometry columns based on the dataset type."""
    if geom_type == "box":
        datasets[["xmin", "ymin", "xmax", "ymax"]] = gpd.GeoSeries.from_wkt(datasets["geometry"]).bounds
    elif geom_type == "point":
        datasets["x"] = gpd.GeoSeries.from_wkt(datasets["geometry"]).centroid.x
        datasets["y"] = gpd.GeoSeries.from_wkt(datasets["geometry"]).centroid.y
    elif geom_type == "polygon":
        datasets["polygon"] = gpd.GeoDataFrame(datasets.geometry).to_wkt()
    return datasets


def create_directories(base_dir, dataset_type):
    """Create directories for the dataset."""
    os.makedirs(f"{base_dir}{dataset_type}_{version}/images", exist_ok=True)
    os.makedirs(f"{base_dir}Mini{dataset_type}_{version}/images", exist_ok=True)


def copy_images(datasets, base_dir, dataset_type):
    """Copy images to the destination folder."""
    for image in datasets["filename"].unique():
        destination = f"{base_dir}{dataset_type}_{version}/images/"
        shutil.copy(image, destination)


def create_mini_datasets(datasets, base_dir, dataset_type, version):
    """Create mini datasets for debugging and generate visualizations."""
    mini_datasets = datasets.groupby("source").first().reset_index(drop=True)
    mini_filenames = mini_datasets["filename"].tolist()
    mini_annotations = datasets[datasets["filename"].isin(mini_filenames)]
    mini_annotations.to_csv(f"{base_dir}Mini{dataset_type}_{version}/official.csv", index=False)
    
    # Copy images for mini datasets
    for image in mini_filenames:
        destination = f"{base_dir}Mini{dataset_type}_{version}/images/"
        shutil.copy(f"{base_dir}{dataset_type}_{version}/images/" + image, destination)

    # Generate visualizations for each source
    for source, group in mini_annotations.groupby("source"):
        group["image_path"] = group["filename"]
        group = read_file(group, root_dir=f"{base_dir}Mini{dataset_type}_{version}/images/")
        group.root_dir = f"{base_dir}Mini{dataset_type}_{version}/images/"
        
        # Remove spaces in source name
        source = source.replace(" ", "_")
        
        # Handle polygons specifically to include image dimensions
        if dataset_type == "TreePolygons":
            height, width, channels = cv2.imread(f"{base_dir}Mini{dataset_type}_{version}/images/" + group.image_path.iloc[0]).shape
            plot_results(group, savedir="/home/b.weinstein/MillionTrees/docs/public/", basename=source, height=height, width=width)
        else:
            plot_results(group, savedir="/home/b.weinstein/MillionTrees/docs/public/", basename=source)

def create_release_files(base_dir, dataset_type):
    """Create release files for the dataset."""
    with open(f"{base_dir}{dataset_type}_{version}/RELEASE_{version}.txt", "w") as outfile:
        outfile.write(f"Version: {version}")

def zip_directory(folder_path, zip_path):
    """Zip the contents of a directory."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

def official_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version):
    """Perform official split and save the results."""
    # Randomly split datasets into train and test (80/20 split)
    TreePolygons_datasets = split_dataset(TreePolygons_datasets, split_column="filename", frac=0.8)
    TreePoints_datasets = split_dataset(TreePoints_datasets, split_column="filename", frac=0.8)
    TreeBoxes_datasets = split_dataset(TreeBoxes_datasets, split_column="filename", frac=0.8)

    # Save the splits to CSV
    TreePolygons_datasets.to_csv(f"{base_dir}TreePolygons_{version}/official.csv", index=False)
    TreePoints_datasets.to_csv(f"{base_dir}TreePoints_{version}/official.csv", index=False)
    TreeBoxes_datasets.to_csv(f"{base_dir}TreeBoxes_{version}/official.csv", index=False)

    print("Official splits saved:")
    print(f"TreePolygons: {base_dir}TreePolygons_{version}/official.csv")
    print(f"TreePoints: {base_dir}TreePoints_{version}/official.csv")
    print(f"TreeBoxes: {base_dir}TreeBoxes_{version}/official.csv")

def cross_geometry_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version):
    """Perform cross-geometry split and save the results."""
    # Assign all polygons to train, points to test, and boxes to test
    TreePolygons_datasets["split"] = "train"
    TreePoints_datasets["split"] = "test"
    TreeBoxes_datasets["split"] = "test"

    # Save the splits to CSV
    TreePolygons_datasets.to_csv(f"{base_dir}TreePolygons_{version}/crossgeometry.csv", index=False)
    TreePoints_datasets.to_csv(f"{base_dir}TreePoints_{version}/crossgeometry.csv", index=False)
    TreeBoxes_datasets.to_csv(f"{base_dir}TreeBoxes_{version}/crossgeometry.csv", index=False)

    print("Cross-geometry splits saved:")
    print(f"TreePolygons: {base_dir}TreePolygons_{version}/crossgeometry.csv")
    print(f"TreePoints: {base_dir}TreePoints_{version}/crossgeometry.csv")
    print(f"TreeBoxes: {base_dir}TreeBoxes_{version}/crossgeometry.csv")

# Zero-shot split
def zero_shot_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version):
    """Perform zero-shot split and save the results."""
    # Define test and train sources
    test_sources_polygons = ["Vasquez et al. 2023", "Miranda et al. 2024"]
    train_sources_polygons = [x for x in TreePolygons_datasets.source.unique() if x not in test_sources_polygons]

    test_sources_points = ["Amirkolaee et al. 2023"]
    train_sources_points = [x for x in TreePoints_datasets.source.unique() if x not in test_sources_points]

    test_sources_boxes = ["Radogoshi et al. 2021"]
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
    TreePolygons_datasets.to_csv(f"{base_dir}TreePolygons_{version}/zeroshot.csv", index=False)
    TreePoints_datasets.to_csv(f"{base_dir}TreePoints_{version}/zeroshot.csv", index=False)
    TreeBoxes_datasets.to_csv(f"{base_dir}TreeBoxes_{version}/zeroshot.csv", index=False)

    print("Zero-shot splits saved:")
    print(f"TreePolygons: {base_dir}TreePolygons_{version}/zeroshot.csv")
    print(f"TreePoints: {base_dir}TreePoints_{version}/zeroshot.csv")
    print(f"TreeBoxes: {base_dir}TreeBoxes_{version}/zeroshot.csv")

def check_for_updated_annotations(dataset, geometry):
    updated_annotations = [pd.read_csv(x) for x in glob.glob(f"data_prep/annotations/*{geometry}*.csv")]
    updated_annotations = pd.concat(updated_annotations)

    dataset["basename"] = dataset["filename"].apply(lambda x: os.path.basename(x))

    # images to remove
    images_to_remove = updated_annotations[updated_annotations.remove =="Remove image from benchmark"].image_path.unique()
    dataset = dataset[~dataset.basename.isin(images_to_remove)]

    updated_annotations = updated_annotations[~(updated_annotations.label.isnull())]

    # Check the filenames
    updated_filenames = updated_annotations["image_path"].unique()
    dataset_filenames = dataset["basename"].unique()

    # Check if any updated filenames are in the dataset
    for filename in updated_filenames:
        if filename in dataset_filenames:
            print(f"Updated annotation found for {filename}")
            
            # Update the dataset with the new annotation
            original_annotations = dataset[dataset["basename"] == filename]
            updated_image_annotations = updated_annotations[updated_annotations["image_path"] == filename].copy(deep=True)
            updated_image_annotations["source"] = original_annotations["source"].values[0]
            updated_image_annotations = read_file(updated_image_annotations, root_dir=os.path.dirname(dataset["filename"].values[0]))
            
            root_dir = os.path.dirname(dataset["filename"].values[0])
            updated_image_annotations["filename"] = updated_image_annotations["image_path"].apply(lambda x: os.path.join(x,root_dir))
            
            # Remove the original annotations
            dataset = dataset[dataset["basename"] != filename]

            # Append the updated annotations
            dataset = pd.concat([dataset, updated_image_annotations], ignore_index=True)
        else:
            continue

def run(version, base_dir, debug=False):
    TreeBoxes = [
        #"/orange/ewhite/DeepForest/Ryoungseob_2023/train_datasets/images/train.csv",
        #"/orange/ewhite/DeepForest/Velasquez_urban_trees/tree_canopies/nueva_carpeta/annotations.csv",
        #'/orange/ewhite/DeepForest/individual_urban_tree_crown_detection/annotations.csv',
        '/orange/ewhite/DeepForest/Radogoshi_Sweden/annotations.csv',
        "/orange/ewhite/DeepForest/WRI/WRI-labels-opensource/annotations.csv",
        #"/orange/ewhite/DeepForest/Guangzhou2022/annotations.csv",
        "/orange/ewhite/DeepForest/NEON_benchmark/NeonTreeEvaluation_annotations.csv",
        "/orange/ewhite/DeepForest/NEON_benchmark/University_of_Florida.csv",
        '/orange/ewhite/DeepForest/ReForestTree/images/train.csv',
        # "/orange/ewhite/DeepForest/Santos2019/annotations.csv"
   ]

    TreePoints = [
        "/orange/ewhite/DeepForest/TreeFormer/all_images/annotations.csv",
        "/orange/ewhite/DeepForest/Ventura_2022/urban-tree-detection-data/images/annotations.csv",
        "/orange/ewhite/MillionTrees/NEON_points/annotations.csv",
        "/orange/ewhite/DeepForest/Tonga/annotations.csv"
    ]

    TreePolygons = [
        "/orange/ewhite/DeepForest/Jansen_2023/pngs/annotations.csv",
        "/orange/ewhite/DeepForest/Troles_Bamberg/coco2048/annotations/annotations.csv",
        "/orange/ewhite/DeepForest/Cloutier2023/images/annotations.csv",
        "/orange/ewhite/DeepForest/Firoze2023/annotations.csv",
        #"/orange/ewhite/DeepForest/Wagner_Australia/annotations.csv",
        #"/orange/ewhite/DeepForest/Alejandro_Chile/alejandro/annotations.csv",
        #"/orange/ewhite/DeepForest/UrbanLondon/annotations.csv",
        #"/orange/ewhite/DeepForest/OliveTrees_spain/Dataset_RGB/annotations.csv",
        #"/orange/ewhite/DeepForest/Araujo_2020/annotations.csv",
        #"/orange/ewhite/DeepForest/justdiggit-drone/label_sample/annotations.csv",
        "/orange/ewhite/DeepForest/BCI/BCI_50ha_2020_08_01_crownmap_raw/annotations.csv",
        "/orange/ewhite/DeepForest/BCI/BCI_50ha_2022_09_29_crownmap_raw/annotations.csv",
        "/orange/ewhite/DeepForest/Harz_Mountains/ML_TreeDetection_Harz/annotations.csv",
        "/orange/ewhite/DeepForest/SPREAD/annotations.csv",
        "/orange/ewhite/DeepForest/KagglePalm/Palm-Counting-349images/annotations.csv",
    ]

    # Combine datasets
    TreeBoxes_datasets = combine_datasets(TreeBoxes, debug=debug)
    TreePoints_datasets = combine_datasets(TreePoints, debug=debug)
    TreePolygons_datasets = combine_datasets(TreePolygons, debug=debug)

    # Remove alpha channels
    remove_alpha_channel(TreeBoxes_datasets)
    remove_alpha_channel(TreePoints_datasets)
    remove_alpha_channel(TreePolygons_datasets)

    # Check for updated annotations
    check_for_updated_annotations(TreeBoxes_datasets, "Boxes")
    check_for_updated_annotations(TreePoints_datasets, "Points")
    #check_for_updated_annotations(TreePolygons_datasets, "Polygons")

    # Split datasets
    TreeBoxes_datasets = split_dataset(TreeBoxes_datasets)
    TreePoints_datasets = split_dataset(TreePoints_datasets)
    TreePolygons_datasets = split_dataset(TreePolygons_datasets)

    # Process geometry columns
    TreeBoxes_datasets = process_geometry_columns(TreeBoxes_datasets, "box")
    TreePoints_datasets = process_geometry_columns(TreePoints_datasets, "point")
    TreePolygons_datasets = process_geometry_columns(TreePolygons_datasets, "polygon")

    # Create directories
    create_directories(base_dir, "TreeBoxes")
    create_directories(base_dir, "TreePoints")
    create_directories(base_dir, "TreePolygons")

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

    # Perform splits
    zero_shot_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version)
    official_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version)
    cross_geometry_split(TreePolygons_datasets, TreePoints_datasets, TreeBoxes_datasets, base_dir, version)

    # Create release files
    create_release_files(base_dir, "TreeBoxes")
    create_release_files(base_dir, "TreePoints")
    create_release_files(base_dir, "TreePolygons")

    # Zip datasets
    zip_directory(f"{base_dir}TreeBoxes_{version}", f"{base_dir}TreeBoxes_{version}.zip")
    zip_directory(f"{base_dir}TreePoints_{version}", f"{base_dir}TreePoints_{version}.zip")
    zip_directory(f"{base_dir}TreePolygons_{version}", f"{base_dir}TreePolygons_{version}.zip")
    zip_directory(f"{base_dir}MiniTreeBoxes_{version}", f"{base_dir}MiniTreeBoxes_{version}.zip")
    zip_directory(f"{base_dir}MiniTreePoints_{version}", f"{base_dir}MiniTreePoints_{version}.zip")
    zip_directory(f"{base_dir}MiniTreePolygons_{version}", f"{base_dir}MiniTreePolygons_{version}.zip")


if __name__ == "__main__":
    version = "v0.1.3.4"
    base_dir = "/orange/ewhite/web/public/"
    debug = False
    run(version, base_dir, debug)