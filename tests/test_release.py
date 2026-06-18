from milliontrees.datasets.TreePolygons import TreePolygonsDataset
from milliontrees.datasets.TreePoints import TreePointsDataset
from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
from milliontrees.common.data_loaders import get_train_loader, get_eval_loader
import torch
import pytest

@pytest.fixture
def dataset_config():
    return {
        "download": False,
        "root_dir": "/orange/ewhite/web/public/MillionTrees",
    }

def _test_dataset_structure(dataset, targets_key="y", expected_shape_check=None):
    """Helper function to test dataset structure consistently"""
    train_dataset = dataset.get_subset("train")
    test_dataset = dataset.get_subset("test")
    
    # Test train subset if it has samples
    if len(train_dataset) > 0:
        # Test single sample
        metadata, image, targets = next(iter(train_dataset))
        y = targets[targets_key]
        labels = targets["labels"]
        
        assert image.shape == (3, 448, 448)
        assert image.dtype == torch.float32
        assert 0.0 <= image.min() and image.max() <= 1.0
        assert metadata.shape[0] == 2
        
        if expected_shape_check:
            expected_shape_check(y)
        
        # Test train loader
        batch_size = min(2, len(train_dataset))
        train_loader = get_train_loader('standard', train_dataset, batch_size=batch_size)
        metadata, x, targets = next(iter(train_loader))
        
        assert len(targets) == batch_size
        assert x.shape == (batch_size, 3, 448, 448)
        assert x.dtype == torch.float32
        assert 0.0 <= x.min() and x.max() <= 1.0
        assert len(metadata) == batch_size
    
    # Test test loader if it has samples
    if len(test_dataset) > 0:
        batch_size = min(2, len(test_dataset))
        test_loader = get_eval_loader('standard', test_dataset, batch_size=batch_size)
        metadata, x, targets = next(iter(test_loader))

        assert len(targets) == batch_size
        assert x.shape == (batch_size, 3, 448, 448)
        assert x.dtype == torch.float32
        assert 0.0 <= x.min() and x.max() <= 1.0
        assert len(metadata) == batch_size
    
    # Ensure at least one split has samples
    assert len(train_dataset) > 0 or len(test_dataset) > 0, "Both train and test subsets are empty"

@pytest.mark.parametrize("split_scheme", ['within-distribution', 'crossgeometry', 'out-of-distribution'])
def test_TreePolygons_latest_release(dataset, split_scheme):
    root_dir = dataset

    dataset = TreePolygonsDataset(
        download=False,
        root_dir=root_dir,
        version="0.0",
        split_scheme=split_scheme,
    )
    
    def check_polygon_shape(y):
        assert y[0].shape == (448, 448)
    
    _test_dataset_structure(dataset, expected_shape_check=check_polygon_shape)

@pytest.mark.parametrize("split_scheme", ['within-distribution', 'crossgeometry', 'out-of-distribution'])
def test_TreePoints_latest_release(dataset, split_scheme):
    dataset = TreePointsDataset(
        download=False,
        root_dir=dataset,
        version="0.0",
        split_scheme=split_scheme,
    )
    
    def check_points_shape(points):
        assert points.shape[1] == 2
    
    _test_dataset_structure(dataset, expected_shape_check=check_points_shape)

@pytest.mark.parametrize("split_scheme", ['within-distribution', 'crossgeometry', 'out-of-distribution'])
def test_TreeBoxes_latest_release(dataset, split_scheme):
    dataset = TreeBoxesDataset(
        download=False,
        root_dir=dataset,
        version="0.0",
        split_scheme=split_scheme,
    )
    
    def check_boxes_shape(boxes):
        assert boxes.shape[1] == 4
    
    _test_dataset_structure(dataset, expected_shape_check=check_boxes_shape)


@pytest.mark.parametrize("split_scheme", ['within-distribution', 'crossgeometry', 'out-of-distribution'])
def test_TreePolygons_small_release(dataset, split_scheme):
    dataset = TreePolygonsDataset(
        download=False,
        root_dir=dataset,
        split_scheme=split_scheme,
        small=True,
        version="0.0",
    )

    def check_polygon_shape(y):
        assert y[0].shape == (448, 448)

    _test_dataset_structure(dataset, expected_shape_check=check_polygon_shape)


@pytest.mark.parametrize("split_scheme", ['within-distribution', 'crossgeometry', 'out-of-distribution'])
def test_TreePoints_small_release(dataset, split_scheme):
    dataset = TreePointsDataset(
        download=False,
        root_dir=dataset,
        split_scheme=split_scheme,
        small=True,
        version="0.0",
    )

    def check_points_shape(points):
        assert points.shape[1] == 2

    _test_dataset_structure(dataset, expected_shape_check=check_points_shape)


@pytest.mark.parametrize("split_scheme", ['within-distribution', 'crossgeometry', 'out-of-distribution'])
def test_TreeBoxes_small_release(dataset, split_scheme):
    dataset = TreeBoxesDataset(
        download=False,
        root_dir=dataset,
        split_scheme=split_scheme,
        small=True,
        version="0.0",
    )

    def check_boxes_shape(boxes):
        assert boxes.shape[1] == 4

    _test_dataset_structure(dataset, expected_shape_check=check_boxes_shape)
