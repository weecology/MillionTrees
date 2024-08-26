from datetime import datetime
from pathlib import Path
import os

from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import torch
from shapely import from_wkt
from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import Accuracy, Recall, F1


class TreePolygonsDataset(MillionTreesDataset):
    """The TreePolygons dataset is a collection of tree annotations annotated
    as multi-point polygons locations.

    The dataset is comprised of many sources from across the world. There are 5 splits:
        - Random: 80% of the data randomly split into train and 20% in test
        - location: 80% of the locations randomly split into train and 20% in test
    Supported `split_scheme`:
        - 'Random'
        - 'location'
    Input (x):
        RGB images from camera traps
    Label (y):
        y is a n x 2-dimensional vector where each line represents a point coordinate (x, y)
    Metadata:
        Each image is annotated with the following metadata
            - location (int): location id
            - source (int): source id
            - resolution (int): resolution of image
            - focal view (int): focal view of image

    Website:
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009180
    Original publication:
        # Ventura et al. 2022
        @article{ventura2022individual,
        title={Individual tree detection in large-scale urban environments using high-resolution multispectral imagery},
        author={Ventura, Jonathan and Pawlak, Camille and Honsberger, Milo and Gonsalves, Cameron and Rice, Julian and Love, Natalie LR and Han, Skyler and Nguyen, Viet and Sugano, Keilana and Doremus, Jacqueline and others},
        journal={arXiv preprint arXiv:2208.10607},
        year={2022}
        }
        # TreeFormer
        #etc....


    License:
        This dataset is distributed under Creative Commons Attribution License
    """
    _dataset_name = 'TreePolygons'
    _versions_dict = {'0.0': {'download_url': 'https://www.dropbox.com/scl/fi/81ost5jvsp7cb8br02mm4/TreePolygons_v0.0.zip?rlkey=cu1u1r6s1qftvedkgl3wo7bji&dl=0', 'compressed_size': '17112645'}}

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='official'):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')
        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load splits
        df = pd.read_csv(self._data_dir / '{}.csv'.format(split_scheme))

        # Splits
        self._split_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
            'id_val': 3,
            'id_test': 4
        }
        self._split_names = {
            'train': 'Train',
            'val': 'Validation (OOD/Trans)',
            'test': 'Test (OOD/Trans)',
            'id_val': 'Validation (ID/Cis)',
            'id_test': 'Test (ID/Cis)'
        }

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df['split_id'].values

        # Filenames
        self._input_array = df['filename'].values

        # Convert each polygon to shapely objects
        for i in range(len(df)):
            df.loc[i, 'polygon'] = from_wkt(df.loc[i, 'polygon'])

        self._y_array = list(df['polygon'].values)

        # Labels -> just 'Tree'
        self._n_classes = 1
        self._y_size = 2

        # Create source locations with a numeric ID
        df["source_id"] = df.source.astype('category').cat.codes

        # Location/group info
        n_groups = max(df['source_id']) + 1
        self._n_groups = n_groups
        assert len(np.unique(df['source_id'])) == self._n_groups

        self._metadata_array = torch.tensor(
            np.stack([df['source_id'].values], axis=1))
        self._metadata_fields = ['source_id']

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(dataset=self,
                                                  groupby_fields=(['source_id'
                                                                  ]))

        super().__init__(root_dir, download, split_scheme)

    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y_polygon = self._y_array[idx]
        y = self.create_polygon_mask(x.shape[-2:], y_polygon)
        metadata = self.metadata_array[idx]

        return x, y, metadata

    def create_polygon_mask(self, image_size, vertices):
        """
        Create a grayscale image with a white polygonal area on a black background.

        Parameters:
        - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
        - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                            of the polygon. Vertices should be in clockwise or counter-clockwise order.

        Returns:
        - PIL.Image.Image: A PIL Image object containing the polygonal mask.
        """
        # Create a new black image with the given dimensions
        mask_img = Image.new('L', image_size, 0)

        # Draw the polygon on the image. The area inside the polygon will be white (255).
        # Get the coordinates of the polygon vertices
        polygon_coords = [(int(vertex[0]), int(vertex[1]))
                          for vertex in vertices.exterior.coords._coords]

        # Draw the polygon on the image. The area inside the polygon will be white (255).
        ImageDraw.Draw(mask_img, 'L').polygon(polygon_coords, fill=(255))

        # Return the image with the drawn polygon as numpy array
        mask_img = np.array(mask_img)

        return mask_img

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """Computes all evaluation metrics.

        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
            })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n")

        return results, results_str

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        # All images are in the images folder
        img_path = os.path.join(self.data_dir / 'images' /
                                self._input_array[idx])
        img = Image.open(img_path)
        # Channels first input
        img = torch.tensor(np.array(img)).permute(2, 0, 1)

        return img
