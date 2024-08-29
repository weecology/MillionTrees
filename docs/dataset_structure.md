# Dataset structure

The organization of this dataset was inspired by the WILDS benchmark and torchgeo python package.
There are three overarching datasets. 'Points', 'Polygons' and 'Boxes' based on the annotation geometry.

## Data download

```
dataset = TreePointsDataset(download=True, root_dir=<directory to save data>) 
```

## Dataloaders

Part of the inspriation of this package is to keep most users from needing to interact with the filesystem. The dataloaders are built in, and for many applications, the user will never need to mess around with csv files or image paths. All annotations are pytorch dataloaders and can be iterated over. There is a 'label' column, but given that it we have just one class, "Tree", it is mostly a convienance.

```
for image, label, metadata in dataset:
    assert image.shape == (3, 100, 100)
    assert label.shape == (2,)
    assert len(metadata) == 2
```

Users can select a subset of the dataset and optionally supply a torchvision transform

```
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
```
## Split Schemes

One of the great things about supplying data as dataloaders is easy access to different ways to combine datasets. The MillionTrees benchmark has multiple tasks, and each of these is a 'split_scheme', following the terminology from the WILDS benchmark. To see supported schemes for each dataset, see the documentation of each dataset, as well as the (leaderboard)[leaderboard.md].

```
dataset = TreePointsDataset(download=True, root_dir=<directory to save data>, split_scheme="official") 
```
This looks at the file official.csv and gets the 'split' column that designates which images are in train/test/val depending on the task.

## Underlying data

If a user does need to inspect the underlying data they will find the following design.

## filename

The filename is the name of the image. All filenames are relative to the data directory. 

## source

The source dataset or author of the images. See See (datasets)[datasets.md] for the component pieces.

## Annotation geometry

### Boxes

Boxes annotations are given as xmin, ymin, xmax, ymax coordinates relative to the image origin (top-left).

### Points

Points annotations are given as x,y coordinate relative to the image origin.

### Polygons

Polygon annotations are given as well-known text coordinates, e.g. "POLYGON((x1 y2, x2 y2, x3, y3 ...))" relative to the image origin.