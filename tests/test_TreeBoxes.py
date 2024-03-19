import os
import sys
sys.path.extend(os.path.dirname(os.path.dirname(__file__)))

def test_TreeBoxes():  
    #Just for the plane
    from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
    dataset = TreeBoxesDataset(download=False, root_dir=os.getcwd()) 
