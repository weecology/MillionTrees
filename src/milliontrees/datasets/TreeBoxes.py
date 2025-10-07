from datetime import datetime
from pathlib import Path
import os

import pandas as pd
import numpy as np
import torch
import albumentations as A
import torchvision.transforms as T
import fnmatch
from types import SimpleNamespace

from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import DetectionAccuracy
from PIL import Image
from milliontrees.download_unsupervised import run as run_unsupervised

from albumentations.pytorch import ToTensorV2


class TreeBoxesDataset(MillionTreesDataset):
    """A dataset of tree annotations with bounding box coordinates from multiple global sources.

    The dataset contains aerial imagery of trees with their corresponding bounding box annotations.
    Each tree is annotated with a 4-point bounding box (x_min, y_min, x_max, y_max).

    Dataset Splits:
        - Random: For each source, 80% of the data is used for training and 20% for testing.
        - crossgeometry: Boxes and Points are used to predict polygons.
        - zeroshot: Selected sources are entirely held out for testing.

    Data Format:
        Input (x): RGB aerial imagery
        Labels (y): Nx4 array of bounding box coordinates
        Metadata: Location identifiers for each image

    Args:
        version (str): The version of the dataset to load.
        root_dir (str): The root directory to store the dataset.
        download (bool): Whether to download the dataset if it is not already present.
        split_scheme (str): The split scheme to use.
        geometry_name (str): The name of the geometry to use.
        eval_score_threshold (float): The threshold for the evaluation score.
        remove_incomplete (bool): Whether to remove incomplete data.
        image_size (int): The size of the image to use.
        include_sources (list): The sources to include.
        exclude_sources (list): The sources to exclude.
        unsupervised (bool): If True, include unsupervised data in addition to
            any other selected sources (unless explicitly excluded).
        unsupervised_args (dict): The arguments to pass to the unsupervised download pipeline.

    References:
        Website: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009180

        Citation:
            @article{Weinstein2020,
            title={A benchmark dataset for canopy crown detection and delineation in co-registered airborne RGB, LiDAR and hyperspectral imagery from the National Ecological Observation Network.},
            author={Weinstein BG, Graves SJ, Marconi S, Singh A, Zare A, Stewart D, et al.},
            journal={PLoS Comput Biol},
            year={2021},
            doi={10.1371/journal.pcbi.1009180}
            }

    License: Creative Commons Attribution License
    """
    _dataset_name = 'TreeBoxes'
    _versions_dict = {
        '0.0': {
            'download_url':
                'https://github.com/weecology/MillionTrees/releases/download/0.0.0-dev1/TreeBoxes_v0.0.zip',
            'compressed_size':
                5940337
        },
        "0.1": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreeBoxes_v0.1.zip",
            'compressed_size':
                3476300
        },
        "0.2": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreeBoxes_v0.2.zip",
            'compressed_size':
                6717977561
        },
        "0.5": {
            'download_url': "",
            'compressed_size': "41108312"
        }
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='random',
                 geometry_name='y',
                 eval_score_threshold=0.1,
                 remove_incomplete=False,
                 image_size=448,
                 include_sources=None,
                 exclude_sources=None,
                 unsupervised=False,
                 unsupervised_args=None):

        self._version = version
        self._split_scheme = split_scheme
        self.geometry_name = geometry_name
        self.eval_score_threshold = eval_score_threshold
        self.image_size = image_size
        self.unsupervised = unsupervised

        if self._split_scheme not in ['random', 'zeroshot', 'crossgeometry']:
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')

        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Load splits
        df = pd.read_csv(self._data_dir / '{}.csv'.format(split_scheme))

        # Optionally trigger unsupervised download pipeline
        if unsupervised:
            # If unsupervised hasn't been downloaded, download it
            if not os.path.exists(
                    self._data_dir /
                    'unsupervised/unsupervised_annotations_tiled..'):
                defaults = {
                    'data_dir':
                        str(self._data_dir),
                    'annotations_parquet':
                        str(self._data_dir /
                            'unsupervised/TreeBoxes_unsupervised.parquet'),
                    'max_tiles_per_site':
                        None,
                    'patch_size':
                        400,
                    'allow_empty':
                        False,
                    'num_workers':
                        4,
                    'token_path':
                        'neon_token.txt',
                    'data_product':
                        'DP3.30010.001',
                    'download_dir':
                        str(self._data_dir / 'unsupervised/neon_downloads'),
                }
                if isinstance(unsupervised_args, dict):
                    defaults.update(unsupervised_args)
                run_unsupervised(**defaults)

        # (moved) Unsupervised data will be appended later, after include/exclude filters,
        # unless it is explicitly excluded.

        # Remove incomplete data based on flag
        if remove_incomplete:
            df = df[df['complete'] == True]

        # Filter by include/exclude source names with wildcard support
        include_patterns = None
        if include_sources is not None and include_sources != []:
            include_patterns = include_sources if isinstance(
                include_sources, (list, tuple)) else [include_sources]
        exclude_patterns = exclude_sources
        if exclude_patterns is None:
            exclude_patterns = []
        elif not isinstance(exclude_patterns, (list, tuple)):
            exclude_patterns = [exclude_patterns]

        source_str = df['source'].astype(str).str.lower()

        if include_patterns is not None:
            patterns_lower = [p.lower() for p in include_patterns]
            mask_include = source_str.apply(
                lambda s: any(fnmatch.fnmatch(s, p) for p in patterns_lower))
            df = df[mask_include]

        patterns_exclude_lower = [p.lower() for p in exclude_patterns]
        if len(patterns_exclude_lower) > 0:
            mask_exclude = source_str.apply(lambda s: any(
                fnmatch.fnmatch(s, p) for p in patterns_exclude_lower))
            df = df[~mask_exclude]

        # After applying filters to labeled data, optionally append unsupervised data additively
        if self.unsupervised:
            unsupervised_dir = self._data_dir / 'unsupervised'
            print(
                f"Loading unsupervised data from {unsupervised_dir}, this may take a while..."
            )
            for root, _, files in os.walk(unsupervised_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.parquet'):
                        unsupervised_df = pd.read_parquet(file_path)
                        unsupervised_df["split"] = "train"
                        df = pd.concat([df, unsupervised_df], ignore_index=True)

        # Splits
        self._split_dict = {
            'train': 0,
            'val': 1,
            'test': 2,
        }
        self._split_names = {
            'train': 'Train',
            'val': 'Validation (OOD/Trans)',
            'test': 'Test (OOD/Trans)',
        }

        unique_files = df.drop_duplicates(subset=['filename'],
                                          inplace=False).reset_index(drop=True)
        unique_files['split_id'] = unique_files['split'].apply(
            lambda x: self._split_dict[x])
        self._split_array = unique_files['split_id'].values

        # Filenames
        self._input_array = unique_files.filename

        # Create lookup table for which index to select for each filename
        self._input_lookup = df.groupby('filename').apply(
            lambda x: x.index.values).to_dict()
        self._y_array = df[["xmin", "ymin", "xmax",
                            "ymax"]].values.astype("float32")

        # Labels -> just 'Tree'
        self._n_classes = 1

        # Length of targets
        self._y_size = 4

        # Class labels
        self.labels = torch.zeros(df.shape[0])

        # Create source locations with a numeric ID
        df["source_id"] = df.source.astype('category').cat.codes

        # Create filename numeric ID
        df["filename_id"] = df.filename.astype('category').cat.codes

        # Create dictionary for codes to names
        self._source_id_to_code = df.set_index('source_id')['source'].to_dict()
        self._filename_id_to_code = df.set_index(
            'filename_id')['filename'].to_dict()

        # Location/group info
        n_groups = max(df['source_id']) + 1
        self._n_groups = n_groups
        assert len(np.unique(df['source_id'])) == self._n_groups

        # Metadata is at the image level
        unique_sources = df[['filename_id', 'source_id']].drop_duplicates(
            subset="filename_id", inplace=False).reset_index(drop=True)
        self._metadata_array = torch.tensor(unique_sources.values.astype('int'))
        self._metadata_fields = ['filename_id', 'source_id']

        self._collate = TreeBoxesDataset._collate_fn

        self.metrics = {
            "accuracy":
                DetectionAccuracy(geometry_name=self.geometry_name,
                                  score_threshold=self.eval_score_threshold,
                                  metric="accuracy"),
            "recall":
                DetectionAccuracy(geometry_name=self.geometry_name,
                                  score_threshold=self.eval_score_threshold,
                                  metric="recall"),
        }

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(dataset=self,
                                                  groupby_fields=(['source_id'
                                                                  ]))

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata):
        """Performs evaluation on the given predictions.

        The main evaluation metric, detection_acc_avg_dom, measures the simple average of the
        detection accuracies of each domain.
        """

        results = {}
        results_str = ''
        for metric in self.metrics:
            result, result_str = self.standard_group_eval(
                self.metrics[metric], self._eval_grouper, y_pred, y_true,
                metadata)
            results[metric] = result
            results_str += result_str

        detection_accs = []
        for k, v in results["accuracy"].items():
            if k.startswith('detection_acc_source:'):
                d = k.split(':')[1]
                count = results["accuracy"][f'source:{d}']
                if count > 0:
                    detection_accs.append(v)
        detection_acc_avg_dom = np.array(detection_accs).mean()
        results['detection_acc_avg_dom'] = detection_acc_avg_dom
        results_str = f'Average detection_acc across source: {detection_acc_avg_dom:.3f}\n' + results_str

        # Format results with tables
        from milliontrees.common.utils import format_eval_results
        formatted_results = format_eval_results(results, self)
        results_str = formatted_results + '\n' + results_str

        return results, results_str

    def get_input(self, idx):
        """Retrieves the input features (image) for a given data point.

        Args:
            idx (int): Index of a data point

        Returns:
            np.ndarray: Input features of the idx-th data point (image) as a normalized numpy array.
        """
        # All images are in the images folder
        img_path = os.path.join(self._data_dir / 'images' /
                                self._input_array[idx])
        img = Image.open(img_path)
        img = np.array(img.convert('RGB')) / 255
        img = np.array(img, dtype=np.float32)

        return img

    @staticmethod
    def _collate_fn(batch):
        """Collates a batch by stacking `x` (features) and `metadata`, but not `y` (targets).

        The batch is initially a tuple of individual data points: (item1, item2, item3, ...).
        After zipping, it transforms into a list of tuples:
        [(item1[0], item2[0], ...), (item1[1], item2[1], ...), ...].

        Args:
            batch (list): A batch of data points, where each data point is a tuple (metadata, x, y).

        Returns:
            tuple: A tuple containing:
                - Stacked `x` (features).
                - Stacked `metadata`.
        """
        batch = list(zip(*batch))
        batch[1] = torch.stack(batch[1])
        batch[0] = torch.stack(batch[0])
        batch[2] = list(batch[2])

        return tuple(batch)

    def _transform_(self):
        transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
            ToTensorV2()
        ],
                              bbox_params=A.BboxParams(format='pascal_voc',
                                                       label_fields=['labels'],
                                                       clip=True))

        return transform
