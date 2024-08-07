from milliontrees.datasets.TreeBoxes import TreeBoxesDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Test structure without real annotation data to ensure format is correct
def test_TreeBoxes_generic(dataset):
    dataset = TreeBoxesDataset(download=False, root_dir=dataset) 
    for image, label, metadata in dataset:
        assert image.shape == (3, 100, 100)
        assert label.shape == (4,)
        # Two fine-grained domain and a label of the coarse domain? This is still unclear see L82 of milliontrees_dataset.py
        assert len(metadata) == 2
        break

# Test structure with real annotation data to ensure format is correct
def test_TreeBoxes_release():
    dataset = TreeBoxesDataset(download=False, root_dir="/orange/ewhite/DeepForest/MillionTrees/")
    train_dataset = dataset.get_subset("train", transform=A.Compose(
        [A.Resize(448, 448), A.HorizontalFlip(p=0.5), ToTensorV2()],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=["category_ids"])
    ))
     
    for image, label, metadata in train_dataset:
        assert image.shape == (3, 448, 448)
        assert label.shape == (4,)
        assert len(metadata) == 2
        break