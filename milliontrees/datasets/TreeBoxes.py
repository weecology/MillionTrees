from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch
import json
import cv2

from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import Accuracy, Recall, F1

class TreeBoxesDataset(MillionTreesDataset):
    """
        The TreeBoxes dataset is a collection of tree annotations annotated as four pointed bounding boxes.
        The dataset is comprised of many sources from across the world. There are 2 splits:
            - Official: 80% of the data randomly split into train and 20% in test
            - Random: 80% of the locations randomly split into train and 20% in test
        Supported `split_scheme`:
            - 'Official'
            - 'Random'
        Input (x):
            RGB images from camera traps
        Label (y):
            y is a n x 4-dimensional vector where each line represents a box coordinate (x_min, y_min, x_max, y_max)
        Metadata:
            Each image is annotated with the following metadata
                - location (int): location id
                - resolution (int): resolution of image
                - focal view (int): focal view of image

        Website:
            https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009180
        Original publication:
            The following publications are included in this dataset
            @article{Weinstein2020,
            title={A benchmark dataset for canopy crown detection and delineation in co-registered airborne RGB, LiDAR and hyperspectral imagery from the National Ecological Observation Network.},
            author={Weinstein BG, Graves SJ, Marconi S, Singh A, Zare A, Stewart D, et al.},
            journal={PLoS Comput Biol},
                    year={2021},
            doi={10.1371/journal.pcbi.1009180}
            }
        Original publication:
            The following publications are included in this dataset
            @article{Weinstein2020,
            title={A benchmark dataset for canopy crown detection and delineation in co-registered airborne RGB, LiDAR and hyperspectral imagery from the National Ecological Observation Network.},
            author={Weinstein BG, Graves SJ, Marconi S, Singh A, Zare A, Stewart D, et al.},
            journal={PLoS Comput Biol},
                    year={2021},
            doi={10.1371/journal.pcbi.1009180}
            }
        License:
            This dataset is distributed under Creative Commons Attribution License
        """
    _dataset_name = 'TreeBoxes'
    _versions_dict = {
        '0.0': {'download_url': '','compressed_size': ""}}


    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme not in ['official','random']:
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load splits
        df = pd.read_csv(self._data_dir / '{}.csv'.format(split_scheme))

        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df['split_id'].values

        # Filenames
        self._input_array = df['filename'].values

        # Box labels
        self._y_array = torch.tensor(df[["xmin", "ymin", "xmax","ymax"]].values.astype(float))
        
        # Labels -> just 'Tree'
        self._n_classes = 1

        # Length of targets
        self._y_size = 4

        # Location/group info
        n_groups = max(df['location']) + 1
        self._n_groups = n_groups
        assert len(np.unique(df['location'])) == self._n_groups

        self._metadata_array = torch.tensor(np.stack([df['location'].values,df['resolution'].values], axis=1))
        self._metadata_fields = ['location','resolution']

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['location']))

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
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
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
        )

        return results, results_str

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """
        # All images are in the images folder
        img_path = os.path.join(self.data_dir / 'images' / self._input_array[idx])
        img = cv2.imread(img_path)
        # Channels first input
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
   

        return img
