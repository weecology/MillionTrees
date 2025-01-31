"""MillionTrees - Benchmark dataset for Airborne Tree Machine Learning"""

__version__ = "0.1.1"
from .get_dataset import get_dataset

benchmark_datasets = [
    'TreePoints',
    'TreeBoxes',
    'TreePolygons',
]

additional_datasets = []

supported_datasets = benchmark_datasets + additional_datasets

unlabeled_datasets = []

unlabeled_splits = []
