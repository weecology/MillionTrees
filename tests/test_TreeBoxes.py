from milliontrees.datasets.TreeBoxes import TreeBoxesDataset

def test_TreeBoxes():
    dataset = TreeBoxesDataset(download=False, root_dir='/blue/ewhite/DeepForest/MillionTrees/') 