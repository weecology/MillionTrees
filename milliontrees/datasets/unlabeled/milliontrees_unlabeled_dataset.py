from torch.utils.data import Dataset

class milliontreesUnlabeledDataset(Dataset):
    """Base class for unlabeled MillionTrees datasets."""
    
    def __init__(self, root_dir, download, split_scheme):
        self.root_dir = root_dir
        
    def initialize_data_dir(self, root_dir, download):
        """Initialize the data directory."""
        return root_dir
        
    @property
    def data_dir(self):
        """The path to the data directory."""
        return self._data_dir 