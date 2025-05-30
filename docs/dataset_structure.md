# Dataset Structure

The organization of this dataset was inspired by the WILDS benchmark and torchgeo python package.
There are three overarching datasets: 'Points', 'Polygons' and 'Boxes' based on the annotation geometry.

## Data download

```python
from milliontrees.datasets.TreePoints import TreePointsDataset
dataset = TreePointsDataset(download=True, root_dir=<directory to save data>) 
```

* Note, even when download=True, if the data already exists in root_dir, the data will not be downloaded a second time.

### Split Schemes

One of the great things about supplying data as dataloaders is easy access to different ways to combine datasets. The MillionTrees benchmark has multiple tasks, and each of these is a 'split_scheme', following the terminology from the WILDS benchmark.

```python
dataset = TreePointsDataset(download=True, root_dir=<directory to save data>, split_scheme="random") 
```

This looks at the file random.csv and gets the 'split' column that designates which images are in train/test/val depending on the task.

The MillionTrees benchmark supports multiple dataset split schemes to accommodate various tasks:

- **Random**: For each source, 80% of the data is used for training, and 20% is reserved for testing.
- **Crossgeometry**: Combines Boxes and Points annotations to predict Polygons.
- **Zeroshot**: Entire sources are held out for testing, simulating a zero-shot learning scenario.

Each split scheme uses the same underlying data, so you don't need to redownload when changing split schemes! 

## Dataset Class

Part of the inspiration of this package is to keep most users from needing to interact with the filesystem. The dataloaders are built in, and for many applications, the user will never need to mess around with csv files or image paths. All annotations are pytorch dataloaders and can be iterated over.

```python
for image, label, metadata in dataset:
    assert image.shape == (3, 100, 100)
    assert label.shape == (2,)
    assert len(metadata) == 2
    break
```


Users can select a subset of the dataset and optionally supply a torchvision transform:

```python
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
```

## Dataloaders

Datasets are batched into lists of target dictionaries, tensors of images, and tensors of metadata.
Each target dictionary contains tensors of the ground truth with the keys dict_keys
(['y', 'labels']). 'y' differs among the TreeGeometry datasets.

```python
train_loader = get_train_loader("standard", train_dataset, batch_size=2)

# Length of the loader, number of batches
len(train_loader)

# Show one batch of the loader
for metadata, image, targets in train_loader:
    print("Targets is a list of dictionaries with the following keys: ", targets[0].keys())
    print(f"Image shape: {image.shape}, Image type: {type(image)}")
    print(f"Annotation shape of the first image: {targets[0]['y'].shape}")
    break
```

### DataFrame Interface
The goal of MillionTrees project is to limit a user's need to access the underyling data structure.
Each dataset maintains a pandas DataFrame containing all annotations and metadata, accessible via the `df` attribute:

```python
dataset = TreePointsDataset()
dataset.df  # Access full DataFrame with annotations and metadata
```

### Incomplete Data Handling
Some datasets may contain incomplete annotations or data quality issues. The `remove_incomplete` flag allows you to filter out these problematic entries:

```python
# Initialize dataset with incomplete data removed
dataset = TreePointsDataset(remove_incomplete=True)
```

When `remove_incomplete=True`, the dataset will:
1. Filter out any annotations marked as incomplete in the source data
2. Only include high-quality, complete annotations in the dataset
3. Apply this filtering to all splits (train/val/test)

This is particularly useful when:
- Working with datasets that have known quality issues
- Training models that require complete, high-quality annotations
- Ensuring consistent data quality across all splits

The DataFrame contains:
- `filename`: Image filename
- `x`, `y`: Point coordinates (TreePoints)
- `xmin`, `ymin`, `xmax`, `ymax`: Box coordinates (TreeBoxes)
- `source`: Original data source
- `split`: Train/test/validation split
- `source_id`: Numeric ID for each source
- `filename_id`: Numeric ID for each image

### Lookup Dictionaries
Helpful mappings between IDs and names:

```python
# Map numeric source IDs to source names
dataset._source_id_to_code  # {0: 'source1', 1: 'source2', ...}

# Map numeric filename IDs to actual filenames
dataset._filename_id_to_code  # {0: 'image1.jpg', 1: 'image2.jpg', ...}

# Map filenames to annotation indices
dataset._input_lookup  # {'image1.jpg': array([0,1,2]), ...}
```

For example, if you want to get the annotations for a specific image, you can use the lookup dictionary:
```
from milliontrees import get_dataset
dataset = get_dataset("TreePoints")
indices = dataset._input_lookup["IMG_904.jpg"]
coordinates = dataset._y_array[indices]
```

## Annotation Geometry

### Boxes
Boxes annotations are given as xmin, ymin, xmax, ymax coordinates relative to the image origin (top-left).

### Points
Points annotations are given as x,y coordinate relative to the image origin.

### Polygons
Polygon annotations are given as well-known text coordinates, e.g. "POLYGON((x1 y2, x2 y2, x3, y3 ...))" relative to the image origin.
