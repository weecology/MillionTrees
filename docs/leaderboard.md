# Leaderboard


# Tasks

There are three tasks within the MillionTrees package. 

## Official

The first task is to create the best global detector for individual trees given a set of training and test data. Datasets are split randomly, reflecting information within localities. This is consistant with how most applied users engage with models, by fine-tuning backbone models with sample data from a desired locality.

## Zero-shot

The second task is to create a zero-shot detection system to generalize across geography and aquisition conditions. Selected datasets are held out from training completely and used for evaluation in new conditions. This is a challenging task with no local training data.

## Cross-geometry

Off the shelf tools often limit users for a single annotation type. We have 'point' models, 'box' models and 'polygon' models. To create truly global models for biological inference, we need models that can use all available data, not just one annotation geometry. In particular, polygon annotations are very time consuming to create, but are often desirable for downstream usecases. We opted against polygon training sources, for example polygons to points, as this is an unrealistic, or atleast, very uncommon downstream use case. 


### Boxes to Polygons

All box sources are used to train and predict all polygon sources. There is no local data from the test localities in train.

### Points to Polygons

All point sources are used to train and predict all polygon sources

### Points to Boxes 

All point sources are used to train and predict all box sources.


# Submissions

## Official

| Name | Citation | Official Split | 
|------|----------|----------------|
|       |            |                  |
|       |            |                  |
|       |            |                  |

## Zero-shot


| Name | Citation | Official Split |
|------|----------|----------------|
|       |            |                  |
|       |            |                  |
|       |            |                  |

## Cross-geometry

### Points to Polygons

### Points to Boxes

### Boxes to Polygons


