from .version import __version__
from .get_dataset import get_dataset

supported_datasets = ['TreePoints', 'TreePolygons', 'TreeBoxes']

benchmark_datasets = [
    'TreePoints',
    'TreeBoxes',
    'TreePolygons',
]

# # Add source completeness and ground sampling distance for each source in each dataset
# for dataset_name in benchmark_datasets:
#     dataset = get_dataset(dataset_name)
#     dataset.add_source_metadata()
