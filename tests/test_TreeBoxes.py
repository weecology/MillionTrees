from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

def test_TreeBoxes(dataset):
    dataset = TreeBoxesDataset(download=False, root_dir=dataset) 
    for image, label, metadata in dataset:
        assert image.shape == (3, 100, 100)
        assert label.shape == (4,)
        # Two fine-grained domain and a label of the coarse domain? This is still unclear see L82 of milliontrees_dataset.py
        assert len(metadata) == 3
        break