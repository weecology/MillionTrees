import pytest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
from PIL import Image

"""
The goal of this module is to create a reproducible example of how datasets are structured. Each has an images folder, and multiple train/test splits.
"""

@pytest.fixture(scope="session")
def dataset():
    # Create a temporary directory
    tmp_dir = tempfile.mkdtemp()

    # Generate the box dataset
    data_dir = os.path.join(tmp_dir, "TreeBoxes_v0.0")
    os.mkdir(data_dir)
    image_dir = os.path.join(data_dir, "images")
    os.mkdir(image_dir)

    box_data = generate_box_dataset(image_dir)
    box_data["split"] = "train"
    box_data.loc[2, "split"] = "test"
    box_file = os.path.join(data_dir, "official.csv")
    box_data.to_csv(box_file)

    second_split = box_data.copy(deep=True)
    second_split['split'] = 'test'
    second_split.loc[2, 'split'] = 'train'
    second_file = os.path.join(image_dir, "random.csv")
    second_split.to_csv(second_file)

    # Save a Release txt file
    release_file = os.path.join(data_dir, "RELEASE_v0.0.txt")
    with open(release_file, "w") as f:
        f.write("v0.0")

    # Generate the polygon dataset
    data_dir = os.path.join(tmp_dir, "TreePolygons_v0.0")
    os.mkdir(data_dir)
    image_dir = os.path.join(data_dir, "images")
    os.mkdir(image_dir)

    polygon_data = generate_polygon_dataset(image_dir)
    polygon_data["split"] = "train"
    polygon_data.loc[2, "split"] = "test"
    polygon_file = os.path.join(data_dir, "official.csv")
    polygon_data.to_csv(polygon_file)

    second_split = polygon_data.copy(deep=True)
    second_split['split'] = 'test'
    second_split.loc[2, 'split'] = 'train'
    second_file = os.path.join(data_dir, "random.csv")
    second_split.to_csv(second_file)

    # Save a Release txt file
    release_file = os.path.join(data_dir, "RELEASE_v0.0.txt")
    with open(release_file, "w") as f:
        f.write("v0.0")
    
    # Generate the point dataset
    data_dir = os.path.join(tmp_dir, "TreePoints_v0.0")
    os.mkdir(data_dir)
    image_dir = os.path.join(data_dir, "images")
    os.mkdir(image_dir)

    point_data = generate_point_dataset(image_dir)
    # Assign each image to a train-test split, make a copy of the dataframe and a new split
    point_data['split'] = 'train'
    point_data.loc[2, 'split'] = 'test'
    point_file = os.path.join(data_dir, "official.csv")
    point_data.to_csv(point_file)

    second_split = point_data.copy(deep=True)
    second_split['split'] = 'test'
    second_split.loc[2, 'split'] = 'train'
    second_file = os.path.join(data_dir, "random.csv")
    second_split.to_csv(second_file)

    # Save a Release txt file
    release_file = os.path.join(data_dir, "RELEASE_v0.0.txt")
    with open(release_file, "w") as f:
        f.write("v0.0")

    return tmp_dir


def generate_box_dataset(image_dir):
    # Generate the box dataset logic here
    # Assuming you have a list of xmin, xmax, ymin, ymax values and corresponding image file paths
    xmin = [10, 20, 30, 15]
    xmax = [50, 60, 70, 55]
    ymin = [15, 25, 35, 20]
    ymax = [55, 65, 75, 60]
    locations = [0, 0, 1, 0]
    resolution = [1, 1, 10, 1]
    image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image1.jpg']

    # Create a pandas DataFrame
    df = pd.DataFrame({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'filename': image_files,"source":locations, "resolution":resolution})

    # Create images and save them to disk within image_dir
    for i, row in df.iterrows():
        # Create a black image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Save the image within image_dir
        image_path = os.path.join(image_dir, row['filename'])
        img = Image.fromarray(img)
        img.save(image_path)

    return df

def generate_polygon_dataset(image_dir):
    # Generate the polygon dataset logic here
    # Assuming you have a list of polygon coordinates and corresponding image file paths
    polygon_wkt = ["POLYGON((10 15, 50 15, 50 55, 10 55, 10 15))", "POLYGON((20 25, 60 25, 60 65, 20 65, 20 25))", "POLYGON((30 35, 70 35, 70 75, 30 75, 30 35))"]
    locations = [0,0,1]
    resolution = [1,1,10]
    image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']

    # Create a pandas DataFrame
    df = pd.DataFrame({'polygon': polygon_wkt, 'filename': image_files, "source":locations,"resolution":resolution})

    # Create images and save them to disk within image_dir
    for i, row in df.iterrows():
        # Create a black image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Save the image within image_dir
        image_path = os.path.join(image_dir, row['filename'])
        img = Image.fromarray(img)
        img.save(image_path)

    return df

def generate_point_dataset(image_dir):
    # Generate the point dataset logic here
    # Assuming you have a list of x, y coordinates and corresponding image file paths
    x = [10, 20, 30]
    y = [15, 25, 35]
    locations = [0,0,1]
    resolution = [1,1,10]
    image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']

    # Create a pandas DataFrame
    df = pd.DataFrame({'x': x, 'y': y, 'filename': image_files,"source":locations,"resolution":resolution})

    # Create images and save them to disk within image_dir
    for i, row in df.iterrows():
        # Create a black image
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Save the image within image_dir
        image_path = os.path.join(image_dir, row['filename'])
        # Convert the numpy array to PIL Image
        pil_img = Image.fromarray(img)

        # Save the image within image_dir
        pil_img.save(image_path)

    return df