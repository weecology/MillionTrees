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
    train_loader = get_train_loader('standard', train_dataset, batch_size=2)
    metadata, x, targets = next(iter(train_loader))
    
    assert len(targets) == 2
    assert x.shape == (2, 3, 448, 448)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert len(metadata) == 2
    
    # Test test loader
    test_dataset = dataset.get_subset("test")
    test_loader = get_eval_loader('standard', test_dataset, batch_size=2)
    metadata, x, targets = next(iter(test_loader))

    assert len(targets) == 2
    assert x.shape == (2, 3, 448, 448)
    assert x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert len(metadata) == 2

@pytest.mark.parametrize("split_scheme", ['random', 'crossgeometry', 'zeroshot'])
def test_TreePolygons_latest_release(tmpdir, dataset_config, split_scheme):
    root_dir = tmpdir if dataset_config["root_dir"] is None else dataset_config["root_dir"]

    dataset = TreePolygonsDataset(
        download=dataset_config["download"], 
        root_dir=root_dir, 
        split_scheme=split_scheme
    )
    
    def check_polygon_shape(y):
        assert y[0].shape == (448, 448)
    
    _test_dataset_structure(dataset, expected_shape_check=check_polygon_shape)

@pytest.mark.parametrize("split_scheme", ['random', 'crossgeometry', 'zeroshot'])
def test_TreePoints_latest_release(tmpdir, dataset_config, split_scheme):
    root_dir = tmpdir if dataset_config["root_dir"] is None else dataset_config["root_dir"]
    
    dataset = TreePointsDataset(
        download=dataset_config["download"], 
        root_dir=root_dir, 
        split_scheme=split_scheme,
    )
    
    def check_points_shape(points):
        assert points.shape[1] == 2
    
    _test_dataset_structure(dataset, expected_shape_check=check_points_shape)

@pytest.mark.parametrize("split_scheme", ['random', 'crossgeometry', 'zeroshot'])
def test_TreeBoxes_latest_release(tmpdir, dataset_config, split_scheme):
    root_dir = tmpdir if dataset_config["root_dir"] is None else dataset_config["root_dir"]
    
    dataset = TreeBoxesDataset(
        download=dataset_config["download"],
        root_dir=root_dir,
        split_scheme=split_scheme
    )
    
    def check_boxes_shape(boxes):
        assert boxes.shape[1] == 4
    
    _test_dataset_structure(dataset, expected_shape_check=check_boxes_shape)
