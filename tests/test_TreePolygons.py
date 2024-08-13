from milliontrees.datasets.TreePolygons import TreePolygonsDataset
import pytest
import os
import torchvision.transforms.v2 as transforms

# Check if running on hipergator
if os.path.exists("/orange"):
    on_hipergator = True
else:
    on_hipergator = False

# Test structure without real annotation data to ensure format is correct
def test_TreePolygons_generic(dataset):
    dataset = TreePolygonsDataset(download=False, root_dir=dataset) 
    for image, label, metadata in dataset:
        assert image.shape == (3, 100, 100)
        assert label.shape == (100, 100)
        assert len(metadata) == 2
        break

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    train_dataset = dataset.get_subset("train", transform=transform)
     
    for image, label, metadata in train_dataset:
        assert image.shape == (3, 448, 448)
        assert label.shape == (448, 448)
        assert len(metadata) == 2
        break

# Test structure with real annotation data to ensure format is correct
# Do not run on github actions
@pytest.mark.skipif(not on_hipergator, reason="Do not run on github actions")
def test_TreePolygons_release():
    dataset = TreePolygonsDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    train_dataset = dataset.get_subset("train", transform=transform)
     
    for image, label, metadata in train_dataset:
        assert image.shape == (3, 448, 448)
        assert label.shape == (4,)
        assert len(metadata) == 2
        break
