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
        sample_indices = np.random.choice(dataset_size, size=min(10, dataset_size), replace=False)
        
        # Get image paths and annotations for sampled indices
        image_paths = []
        annotations = []
        for i in sample_indices:
            data = train_dataset[i]
            filename_id = data[0][0]
            image_path = dataset._filename_id_to_code[int(filename_id)]
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
        upload_to_label_studio(
            images=image_paths,
            sftp_client=sftp_client,
            dataset_type=dataset_type,
            url=os.getenv("LABEL_STUDIO_URL"), 
            project_name=f"MillionTrees-{dataset_type}",
            images_to_annotate_dir=dataset._data_dir / "images" ,
            folder_name=os.getenv("LABEL_STUDIO_DATA_DIR"),
            preannotations=annotations,
            batch_size=10
        )

if __name__ == "__main__":
    main() 