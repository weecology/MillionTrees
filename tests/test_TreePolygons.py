from milliontrees.datasets.TreePolygons import TreePolygonsDataset

def test_TreePolygons(dataset):
    dataset = TreePolygonsDataset(download=False, root_dir=dataset) 