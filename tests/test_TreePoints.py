from milliontrees.datasets.TreePoints import TreePointsDataset

def test_TreePoints(dataset):
    dataset = TreePointsDataset(download=False, root_dir=dataset)
