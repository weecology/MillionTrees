from pathlib import Path
import numpy as np
from milliontrees import get_dataset
import pandas as pd
import os

from label_studio_utils import (
    connect_to_label_studio,
    create_sftp_client,
    upload_to_label_studio,
    check_for_new_annotations,
)

# Label configs by dataset type
LABEL_CONFIGS = {
    'TreeBoxes': """
        <View>
          <Image name="image" value="$image"/>
          <RectangleLabels name="label" toName="image">
            <Label value="tree" background="green" selected="true"/>
          </RectangleLabels>
          <Choices name="complete" toName="image" required="true">
            <Choice value="yes" selected="true">Image is completely annotated</Choice>
            <Choice value="no">Image needs more annotations</Choice>
          </Choices>
          <Choices name="remove" toName="image" required="true">
            <Choice value="yes">Remove image from benchmark</Choice>
            <Choice value="no" selected="true">Keep image in benchmark</Choice>
          </Choices>
        </View>
    """,
    'TreePoints': """
        <View>
          <Image name="image" value="$image"/>
          <KeyPointLabels name="label" toName="image">
            <Label value="tree" background="green" selected="true"/>
          </KeyPointLabels>
          <Choices name="complete" toName="image" required="true">
            <Choice value="yes" selected="true">Image is completely annotated</Choice>
            <Choice value="no">Image needs more annotations</Choice>
          </Choices>
          <Choices name="remove" toName="image" required="true">
            <Choice value="yes">Remove image from benchmark</Choice>
            <Choice value="no" selected="true">Keep image in benchmark</Choice>
          </Choices>
        </View>
    """,
    'TreePolygons': """
        <View>
          <Image name="image" value="$image"/>
          <PolygonLabels name="label" toName="image">
            <Label value="tree" background="green" selected="true"/>
          </PolygonLabels>
          <Choices name="complete" toName="image" required="true">
            <Choice value="yes" selected="true">Image is completely annotated</Choice>
            <Choice value="no">Image needs more annotations</Choice>
          </Choices>
          <Choices name="remove" toName="image" required="true">
            <Choice value="yes">Remove image from benchmark</Choice>
            <Choice value="no" selected="true">Keep image in benchmark</Choice>
          </Choices>
        </View>
    """
}

def _load_test_records(base_dir: str, version: str, dataset_type: str, split_name: str) -> pd.DataFrame:
    """Load split CSV and return only test rows (if split column exists)."""
    csv_path = Path(base_dir) / f"{dataset_type}_{version}" / f"{split_name}.csv"
    df = pd.read_csv(csv_path)
    return df[df["split"].eq("test")] if "split" in df.columns else df

def _build_preannotations(df: pd.DataFrame, dataset_type: str, images_root: Path, select_filenames: list[str]) -> tuple[list[Path], list[pd.DataFrame]]:
    """Create image paths and per-image DataFrames as preannotations for upload."""
    df = df[df["filename"].isin(select_filenames)].copy()
    images: list[Path] = [images_root / fname for fname in sorted(df["filename"].unique())]
    preannotations: list[pd.DataFrame] = []
    for fname in sorted(df["filename"].unique()):
        rows = df[df["filename"] == fname].copy()
        if dataset_type == "TreePoints":
            cols = [c for c in ("x", "y") if c in rows.columns]
            rows = rows[cols].copy()
            rows.insert(0, "image_path", fname)
            rows["label"] = "Tree"
        elif dataset_type == "TreeBoxes":
            cols = [c for c in ("xmin", "ymin", "xmax", "ymax") if c in rows.columns]
            rows = rows[cols].copy()
            rows.insert(0, "image_path", fname)
            rows["label"] = "Tree"
        else:
            if "geometry" in rows.columns:
                rows = rows[["geometry"]].copy()
                rows.insert(0, "image_path", fname)
                rows["label"] = "Tree"
        preannotations.append(rows.reset_index(drop=True))
    return images, preannotations

def upload_eval_splits(version: str, base_dir: str = "/orange/ewhite/web/public/MillionTrees/", num_images: int = 100) -> None:
    """Upload up to `num_images` test images per dataset/split with preannotations to Label Studio.
    
    Projects created: MillionTrees-Eval-<dataset_type>-<split>
    """
    images_to_annotate_dir = Path("data_prep") / "images_to_annotate"
    annotated_images_dir = Path("data_prep") / "images_annotated"
    csv_dir = Path("data_prep") / "annotations"
    for d in (images_to_annotate_dir, annotated_images_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    sftp_client = create_sftp_client(
        user=os.getenv("USER"),
        host=os.getenv("HOST"),
        key_filename=os.path.expanduser(os.getenv("KEY_FILENAME"))
    )

    for dataset_type in ("TreeBoxes", "TreePoints"):
        for split_name in ("random", "zeroshot"):
            df = _load_test_records(base_dir, version, dataset_type, split_name)
            if df.empty:
                print(f"No test records for {dataset_type} {split_name}, skipping.")
                continue
            filenames = df["filename"].dropna().unique().tolist()
            if not filenames:
                print(f"No filenames found for {dataset_type} {split_name}, skipping.")
                continue
            selected = filenames[:min(num_images, len(filenames))]
            images_root = Path(base_dir) / f"{dataset_type}_{version}" / "images"
            images, preannotations = _build_preannotations(df, dataset_type, images_root, selected)

            project_name = f"MillionTrees-Eval-{dataset_type}-{split_name}"
            _ = connect_to_label_studio(
                url=os.getenv("LABEL_STUDIO_URL"),
                project_name=project_name,
                label_config=LABEL_CONFIGS[dataset_type]
            )
            upload_to_label_studio(
                images=images,
                sftp_client=sftp_client,
                dataset_type=dataset_type,
                url=os.getenv("LABEL_STUDIO_URL"),
                project_name=project_name,
                images_to_annotate_dir=images_root,
                folder_name=os.getenv("LABEL_STUDIO_DATA_DIR"),
                preannotations=preannotations,
                batch_size=10
            )
            print(f"Uploaded {len(images)} {dataset_type} {split_name} images to {project_name}.")

def download_eval_annotations(version: str, base_dir: str = "/orange/ewhite/web/public/MillionTrees/") -> None:
    """Download completed eval annotations into data_prep/annotations for all dataset/split combos."""
    images_to_annotate_dir = Path("data_prep") / "images_to_annotate"
    annotated_images_dir = Path("data_prep") / "images_annotated"
    csv_dir = Path("data_prep") / "annotations"
    for d in (images_to_annotate_dir, annotated_images_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    sftp_client = create_sftp_client(
        user=os.getenv("USER"),
        host=os.getenv("HOST"),
        key_filename=os.path.expanduser(os.getenv("KEY_FILENAME"))
    )

    for dataset_type in ("TreeBoxes", "TreePoints"):
        for split_name in ("random", "zeroshot"):
            project_name = f"MillionTrees-Eval-{dataset_type}-{split_name}"
            print(f"Checking for completed annotations: {project_name}")
            _ = check_for_new_annotations(
                sftp_client=sftp_client,
                url=os.getenv("LABEL_STUDIO_URL"),
                project_name=project_name,
                csv_dir=csv_dir,
                images_to_annotate_dir=images_to_annotate_dir,
                annotated_images_dir=annotated_images_dir,
                folder_name=os.getenv("LABEL_STUDIO_DATA_DIR"),
                dataset_type=dataset_type
            )

def format_annotations(dataset_type, boxes, image_path):
    """Format annotations based on dataset type"""
    if dataset_type == 'TreePoints':
        # Points are stored as x,y coordinates in numpy array
        points = boxes # boxes is a numpy array of shape (N,2) containing x,y coordinates
        labels = ['Tree'] * len(points)
        
        # Create dataframe with x,y coordinates and labels
        df = pd.DataFrame({
            'image_path': [image_path] * len(points),
            'x': points[:,0], # Get x coordinates from first column
            'y': points[:,1], # Get y coordinates from second column 
            'label': labels
        })
        return df
        
    elif dataset_type == 'TreeBoxes':
        # Boxes are stored as xmin,ymin,xmax,ymax in numpy array
        # boxes is a numpy array of shape (N,4) containing box coordinates
        labels = ['Tree'] * len(boxes)
        # Create dataframe with box coordinates and labels
        df = pd.DataFrame({
            'image_path': image_path,
            'xmin': boxes[:,0],
            'ymin': boxes[:,1], 
            'xmax': boxes[:,2],
            'ymax': boxes[:,3],
            'label': labels
        })
        return df
        
    elif dataset_type == 'TreePolygons':
        # Polygons are stored as list of x,y coordinates
        polygons = boxes
        labels = ['Tree']
        
        # Create dataframe with polygon coordinates and labels
        df = pd.DataFrame({
            'image_path': image_path,
            'geometry': polygons,
            'label': labels
        })
        return df

def main():
    # Create local directories if they don't exist
    base_dir = Path("data_prep")
    images_to_annotate_dir = base_dir / "images_to_annotate"
    annotated_images_dir = base_dir / "images_annotated"
    csv_dir = base_dir / "annotations"
    
    for directory in [images_to_annotate_dir, annotated_images_dir, csv_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Create SFTP client
    sftp_client = create_sftp_client(
        user=os.getenv("USER"),
        host=os.getenv("HOST"), 
        key_filename=os.path.expanduser(os.getenv("KEY_FILENAME"))
    )
    
    # Process each dataset
    for dataset_type in ['TreeBoxes', 'TreePoints']:
        print(f"\nProcessing {dataset_type}...")
        
        # Initialize Label Studio client with appropriate config
        client = connect_to_label_studio(
            url=os.getenv("LABEL_STUDIO_URL"),
            project_name=f"MillionTrees-{dataset_type}",
            label_config=LABEL_CONFIGS[dataset_type]
        )
        
        # Get dataset
        dataset = get_dataset(dataset_type, download=True)
        train_dataset = dataset.get_subset("train")

        # Sample random images
        dataset_size = len(train_dataset)
        sample_indices = np.random.choice(dataset_size, size=min(200, dataset_size), replace=False)
        
        # Get image paths and annotations for sampled indices
        image_paths = []
        annotations = []
        for i in sample_indices:
            data = train_dataset[i]
            filename_id = data[0][0]
            image_path = dataset._filename_id_to_code[int(filename_id)]
            # If ends in .tif, skip
            if image_path.endswith(".tif"):
                continue
            indices = dataset._input_lookup[image_path]
            boxes = dataset._y_array[indices,:]
            full_path = dataset._data_dir / "images" / image_path
            image_paths.append(full_path)  # Image path
            annotations.append(format_annotations(dataset_type, boxes, image_path))
        
        # Check for completed annotations and download them
        new_annotations = check_for_new_annotations(
            sftp_client=sftp_client,
            url=os.getenv("LABEL_STUDIO_URL"),
            project_name=f"MillionTrees-{dataset_type}",
            csv_dir=csv_dir,
            images_to_annotate_dir=images_to_annotate_dir,
            annotated_images_dir=annotated_images_dir,
            folder_name=os.getenv("LABEL_STUDIO_DATA_DIR"),
            dataset_type=dataset_type
        )

        # Upload new batch of images
        # upload_to_label_studio(
        #     images=image_paths,
        #     sftp_client=sftp_client,
        #     dataset_type=dataset_type,
        #     url=os.getenv("LABEL_STUDIO_URL"), 
        #     project_name=f"MillionTrees-{dataset_type}",
        #     images_to_annotate_dir=dataset._data_dir / "images" ,
        #     folder_name=os.getenv("LABEL_STUDIO_DATA_DIR"),
        #     preannotations=annotations,
        #     batch_size=10
        # )

if __name__ == "__main__":
    #main() 
    upload_eval_splits(version="v0.9", base_dir="/orange/ewhite/web/public/MillionTrees/", num_images=100)
