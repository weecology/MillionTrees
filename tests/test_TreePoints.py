from milliontrees.datasets.TreePoints import TreePointsDataset

# Test structure without real annotation data to ensure format is correct
def test_TreePoints_generic(dataset):
    dataset = TreePointsDataset(download=False, root_dir=dataset) 
    for image, label, metadata in dataset:
        assert image.shape == (3, 100, 100)
        assert label.shape == (2,)
        # Two fine-grained domain and a label of the coarse domain? This is still unclear see L82 of milliontrees_dataset.py
        assert len(metadata) == 3
        break

# Test structure with real annotation data to ensure format is correct
def test_TreePoints_release():
    dataset = TreePointsDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/") 
    for image, label, metadata in dataset:
        assert image.shape == (3, 100, 100)
        assert label.shape == (2,)
        # Two fine-grained domain and a label of the coarse domain? This is still unclear see L82 of milliontrees_dataset.py
        assert len(metadata) == 3
        break