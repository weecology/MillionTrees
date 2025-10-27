from .version import __version__
from .get_dataset import get_dataset

# Optional Hugging Face integration
try:
    from .hf_loader import HuggingFaceLoader, list_available_datasets
    HF_AVAILABLE = True
except (ImportError, NameError):
    HF_AVAILABLE = False

benchmark_datasets = [
    'TreePoints',
    'TreeBoxes',
    'TreePolygons',
]

# Add source completeness and ground sampling distance for each source in each dataset
for dataset_name in benchmark_datasets:
    dataset = get_dataset(dataset_name)
    dataset.add_source_metadata()