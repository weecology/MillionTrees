from datetime import datetime
from pathlib import Path
import os

from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from shapely import from_wkt
from milliontrees.datasets.milliontrees_dataset import MillionTreesDataset
from milliontrees.common.grouper import CombinatorialGrouper
from milliontrees.common.metrics.all_metrics import MaskAccuracy
from torchvision.tv_tensors import BoundingBoxes, Mask
from torchvision.ops import masks_to_boxes
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import fnmatch
from types import SimpleNamespace
from milliontrees.download_unsupervised import run as run_unsupervised


class TreePolygonsDataset(MillionTreesDataset):
    """The TreePolygons dataset is a collection of tree annotations annotated as multi-point polygon
    locations.

    The dataset is comprised of many sources from across the world.

    Dataset Splits:
        - Random: For each source, 80% of the data is used for training and 20% for testing.
        - crossgeometry: Boxes and Points are used to predict polygons.
        - zeroshot: Selected sources are entirely held out for testing.

    Input (x):
        RGB images from camera traps.

    Label (y):
        y is an n x 2-dimensional vector where each line represents a point coordinate (x, y).

    Metadata:
        Each image is annotated with the following metadata:
            - location (int): location id
            - source (int): source id

    License:
        This dataset is distributed under the Creative Commons Attribution License.
    """
    _dataset_name = 'TreePolygons'
    _versions_dict = {
        '0.0': {
            'download_url':
                'https://github.com/weecology/MillionTrees/releases/download/0.0.0-dev1/TreePolygons_v0.0.zip',
            'compressed_size':
                17112645
        },
        "0.1": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.1.zip",
            'compressed_size':
                40277152
        },
        "0.2": {
            'download_url':
                "https://data.rc.ufl.edu/pub/ewhite/MillionTrees/TreePolygons_v0.2.zip",
            'compressed_size':
                75419767345
        },
        "0.5": {
            'download_url': "",
            'compressed_size': "108369576"
        }
    }

    def __init__(self,
                 version=None,
                 root_dir='data',
                 download=False,
                 split_scheme='random',
                 geometry_name='y',
                 eval_score_threshold=0.5,
                 image_size=448,
                 remove_incomplete=False,
                 include_sources=None,
                 exclude_sources=None,
                 unsupervised=False,
                 unsupervised_args=None):

        self._version = version
        self._split_scheme = split_scheme
        self.geometry_name = geometry_name
        self.image_size = image_size
        self.eval_score_threshold = eval_score_threshold

        if self._split_scheme != 'random':
            raise ValueError(
                f'Split scheme {self._split_scheme} not recognized')
        # path
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))

        # Optionally trigger unsupervised download pipeline
        if unsupervised:
            defaults = {
                'data_dir':
                    str(self._data_dir),
                'annotations_parquet':
                    self._data_dir /
                    'unsupervised/TreePolygons_unsupervised.parquet',
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
                    'neon_downloads',
            }
            if isinstance(unsupervised_args, dict):
                defaults.update(unsupervised_args)
            run_unsupervised(**defaults)

        # Load splits
        df = pd.read_csv(self._data_dir / '{}.csv'.format(split_scheme))

        # Load unsupervised data if it is included or not excluded
        if (include_sources and any('unsupervised' in src for src in include_sources)) or \
           (exclude_sources and not any('unsupervised' in src for src in exclude_sources)):
            unsupervised_dir = self._data_dir / 'unsupervised'
            print(
                f"Loading unsupervised data from {unsupervised_dir}, this may take a while..."
            )
            for root, _, files in os.walk(unsupervised_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if file.endswith('.parquet'):
                            unsupervised_df = pd.read_parquet(file_path)
                        elif file.endswith('.csv'):
                            unsupervised_df = pd.read_csv(file_path)
                        else:
                            continue
                        df = pd.concat([df, unsupervised_df], ignore_index=True)
                    except Exception as e:
                        print(f"Warning: failed to read {file_path}: {e}")

        # Remove incomplete data based on flag
        if remove_incomplete:
            df = df[df['complete'] == True]

        # Filter by include/exclude source names with wildcard support
        # Default: exclude sources containing 'unsupervised'
        include_patterns = None
        if include_sources is not None and include_sources != []:
            include_patterns = include_sources if isinstance(
                include_sources, (list, tuple)) else [include_sources]
        exclude_patterns = exclude_sources
        if exclude_patterns is None:
            exclude_patterns = ['*unsupervised*']
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

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df['split_id'].values

        # Filenames
        self._input_array = unique_files.filename

        # Create lookup table for which index to select for each filename
        self._input_lookup = df.groupby('filename').apply(
            lambda x: x.index.values).to_dict()

        # Convert each polygon to shapely objects
        df['polygon'] = df['polygon'].apply(from_wkt)

        self._y_array = list(df['polygon'].values)

        # Labels -> just 'Tree'
        self._n_classes = 1

        # Not clear what this is, since we have a polygon, unknown size
        self._y_size = 4

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

        # eval grouper
        self.metrics = {
            "accuracy":
                MaskAccuracy(geometry_name=self.geometry_name,
                             score_threshold=self.eval_score_threshold,
                             metric="accuracy"),
            "recall":
                MaskAccuracy(geometry_name=self.geometry_name,
                             score_threshold=self.eval_score_threshold,
                             metric="recall"),
        }
        self._eval_grouper = CombinatorialGrouper(dataset=self,
                                                  groupby_fields=(['source_id'
                                                                  ]))

        super().__init__(root_dir, download, split_scheme)

    def __getitem__(self, idx):
        """
        Args:
            - idx (int): Index of a data point

        Output:
            - metadata (Tensor): Metadata of the idx-th data point
            - x (np.ndarray): Input features of the idx-th data point
            - targets (dict): Dictionary containing:
                - "y" (Tensor): Masks of the polygons
                - "bboxes" (BoundingBoxes): Bounding boxes of the polygons
                - "labels" (np.ndarray): Labels for each mask (all zeros in this case)
        """
        x = self.get_input(idx)
        y_indices = self._input_lookup[self._input_array[idx]]
        y_polygons = [self._y_array[i] for i in y_indices]
        mask_imgs = [
            self.create_polygon_mask(x.shape[:2], y_polygon)
            for y_polygon in y_polygons
        ]
        masks = torch.stack([Mask(mask_img) for mask_img in mask_imgs])
        bboxes = BoundingBoxes(data=masks_to_boxes(masks),
                               format='xyxy',
                               canvas_size=x.shape[:2])
        masks = np.stack([mask.numpy() for mask in masks])

        metadata = self._metadata_array[idx]
        targets = {
            "y": masks,
            "bboxes": bboxes,
            "labels": np.zeros(len(masks), dtype=int)
        }

        return metadata, x, targets

    def create_polygon_mask(self, image_size, vertices):
        """Create a grayscale image with a white polygonal area on a black background.

        Parameters:
        - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
        - vertices (shapely.geometry.Polygon): A shapely Polygon object representing the polygon.

        Returns:
        - mask_img (np.ndarray): A numpy array representing the image with the drawn polygon.
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

    def eval(self, y_pred, y_true, metadata):
        """The main evaluation metric, detection_acc_avg_dom, measures the simple average of the
        detection accuracies of each domain."""

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
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (np.ndarray): Input features of the idx-th data point
        """
        # All images are in the images folder
        img_path = os.path.join(self._data_dir / 'images' /
                                self._input_array[idx])
        img = Image.open(img_path)
        img = np.array(img.convert('RGB')) / 255
        img = np.array(img, dtype=np.float32)

        return img

    def _transform_(self):
        transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
            ToTensorV2()
        ],
                              bbox_params=A.BboxParams(format='pascal_voc',
                                                       label_fields=['labels'],
                                                       clip=True))

        return transform
