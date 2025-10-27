# Datasets
There are three datasets within the MillionTrees package: TreeBoxes, TreePoints, and TreePolygons. These datasets contain many source datasets from dozens of papers and research efforts. Below, each source is briefly described. The images for each dataset are generated directly from the dataloaders to allow rapid verification of the annotation status and are regenerated automatically when a new dataset is released or updated. Sample images within a dataset are *randomly selected* within  to ensure transparency. This does mean that some images are sparsely annotated compared to the rest of the images in a source.

*Note: The datasets below are processed and will be part of the final release. The current release is pre-release and not final. Only publically available datasets are included at this time.*

## Dataset Filtering and Management

MillionTrees datasets can contain millions of annotations. Use filtering capabilities to manage dataset size and preview data before downloading.

### Source Filtering

List and filter available sources:

```py
dataset = TreePointsDataset(version="0.5", download=False)
sources = dataset.get_available_sources()
print("Available sources:", sources)
# Available sources: ['Kattenborn_NewZealand', 'NeonTreeEvaluation', 'OFO_unsupervised', 'NEON_unsupervised'...]
```

Include only specific sources:

```py
dataset = TreePointsDataset(
  version="0.5",
  include_sources=['NeonTreeEvaluation', 'OFO_unsupervised']
)
```

Exclude specific sources (exact names or glob patterns supported):

```py
# Exclude a single source by name
dataset = TreePointsDataset(version="0.5", exclude_sources=['NEON_unsupervised'])

# Exclude by pattern (wildcards)
dataset = TreePointsDataset(version="0.5", exclude_sources=['*_unsupervised'])
```

### Size Management and Preview

Preview datasets before downloading to understand their size:

```py
# Preview full dataset
dataset = TreeBoxesDataset(version="0.2", preview_only=True)

# Output shows:
# Total boxes: 2,456,789
# Unique sources: 28
# Boxes by source:
#   NEON_unsupervised: 2,000,000 boxes (81.4%)
#   NeonTreeEvaluation: 200,000 boxes (8.1%)
#   ...
```

Control dataset size with filtering parameters:

```py
# Limit samples per source to balance training data
dataset = TreeBoxesDataset(
    version="0.2",
    max_samples_per_source=5000,  # Max 5000 samples per source
    preview_only=True
)

# Random sampling for computational constraints
dataset = TreeBoxesDataset(
    version="0.2", 
    sample_fraction=0.1,  # Use 10% of available data
    preview_only=True
)

# Combined filtering for fine control
dataset = TreeBoxesDataset(
    version="0.2",
    max_samples_per_source=1000,  # Limit per source
    sample_fraction=0.05,         # Then 5% sample
    min_samples_per_source=50,    # Exclude small sources
    preview_only=True
)
```

### Recommended Workflow

1. **Preview first**: Use `preview_only=True` to understand dataset size
2. **Design filtering**: Choose parameters based on your computational resources
3. **Preview with filtering**: Verify the filtered dataset size
4. **Download**: Use same parameters with `preview_only=False`

```py
# Step 1: Preview full dataset
full_preview = TreeBoxesDataset(version="0.2", preview_only=True)

# Step 2: Design filtering based on preview
filtered_preview = TreeBoxesDataset(
    version="0.2",
    max_samples_per_source=1000,
    sample_fraction=0.1,
    preview_only=True
)

# Step 3: Download with same parameters
dataset = TreeBoxesDataset(
    version="0.2",
    max_samples_per_source=1000,
    sample_fraction=0.1,
    download=True
)
```

### Common Filtering Strategies

- **Prototyping**: `max_samples_per_source=100, sample_fraction=0.01` (very small, fast)
- **Balanced training**: `max_samples_per_source=5000` (prevent source dominance)
- **Resource-constrained**: `sample_fraction=0.1` (10% of full dataset)
- **Large-scale training**: `max_samples_per_source=50000` (large but manageable)

# Boxes

## Dumortier 2025

![sample_image](public/Dumortier_et_al._2025.png)

Cite: 10.5281/zenodo.15155080

https://zenodo.org/records/15155081

## Kaggle Palm Counting

### Source Name: "Kaggle_Palm_Counting"

https://www.kaggle.com/datasets/praneethsaikolla/palm-tree-detection-dataset

![sample_image](public/Kaggle_Palm_Counting.png)

## Kwon et al. 2023

![sample_image](public/Kwon_et_al._2023.png)

**Citation:** Ryoungseob Kwon, Youngryel Ryu, Tackang Yang, Zilong Zhong, Jungho Im,  
*Merging multiple sensing platforms and deep learning empowers individual tree mapping and species detection at the city scale*,  
ISPRS Journal of Photogrammetry and Remote Sensing, Volume 206, 2023

**Location:** Suwon, South Korea

## Radogoshi et al. 2021 

![sample_image](public/Radogoshi_et_al._2021.png)

**Link:** [https://lila.science/datasets/forest-damages-larch-casebearer/](https://lila.science/datasets/forest-damages-larch-casebearer/)

**Location:** Sweden

## Sun et al. 2022

![sample_image](public/Sun_et_al._2022.png)

**Link:** [https://www.sciencedirect.com/science/article/pii/S030324342100369X](https://www.sciencedirect.com/science/article/pii/S030324342100369X)

## Santos et al. 2019

![sample_image](public/Santos_et_al._2019.png)

**Link:** [Dataset Ninja](https://datasetninja.com/tree-species-detection)  

**Citation**

@Article{s19163595,
  AUTHOR = {Santos, Anderson Aparecido dos and Marcato Junior, José and Araújo, Márcio Santos and Di Martini, David Robledo and Tetila, Everton Castelão and Siqueira, Henrique Lopes and Aoki, Camila and Eltner, Anette and Matsubara, Edson Takashi and Pistori, Hemerson and Feitosa, Raul Queiroz and Liesenberg, Veraldo and Gonçalves, Wesley Nunes},
  TITLE = {Assessment of CNN-Based Methods for Individual Tree Detection on Images Captured by RGB Cameras Attached to UAVs},
  JOURNAL = {Sensors},
  VOLUME = {19},
  YEAR = {2019},
  NUMBER = {16},
  ARTICLE-NUMBER = {3595},
  URL = {https://www.mdpi.com/1424-8220/19/16/3595},
DOI = {10.3390/s19163595}
}

**Location:** Barro Colorado Island, Panama

## Velasquez-Camacho et al. 2023

![sample_image](public/Velasquez-Camacho_et_al._2023.png)

**Link:** [https://zenodo.org/records/10246449](https://zenodo.org/records/10246449)

**Location:** Spain

## Weinstein et al. 2021

![sample_image](public/NEON_benchmark.png)

**Link:** https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009180

**Location** [NEON sites](https://www.neonscience.org/field-sites/explore-field-sites) within the United States

An extension of this published resource was made by the Weecology Lab at the University of Florida

![sample_image](public/Weecology_University_Florida.png)

## World Resources Institute 

NAIP Imagery from across the United States

![sample_image](public/World_Resources_Institute.png)

## SelvaBox (CanopyRS)

![sample_image](public/SelvaBox.png)

**Citation:** SelvaBox: A high-resolution dataset for tropical tree crown detection in dense canopies  
**Link:** [https://huggingface.co/datasets/CanopyRS/SelvaBox](https://huggingface.co/datasets/CanopyRS/SelvaBox)  
**Location:** Brazil, Ecuador, and Panama  
**Description:** High-resolution dataset (4.5 cm GSD) with over 83,000 human bounding box annotations for tropical tree crowns in dense canopies. The dataset comprises 14 rasters from three different countries with varying spatial extents.

## Veitch-Michaelis et al. 2024.

![sample_image](public/OAM-TCD.png)

**Link:** [HuggingFace Dataset](https://huggingface.co/datasets/restor/tcd)

https://zenodo.org/records/11617167


**Location:** Global

**Citation:**  Veitch-Michaelis, J., Cottam, A., Schweizer, D., Broadbent, E., Dao, D., Zhang, C., Almeyda Zambrano, A., & Max, S. (2024). OAM-TCD: A globally diverse dataset of high-resolution tree cover maps (1.0.0) [Data set]. Zenodo.

For more information about the dataset collation, see: Veitch-Michaelis, J. et al. "OAM-TCD: A globally diverse dataset of high-resolution tree cover maps." Advances in neural information processing systems 37 (2024): 49749-49767.

## Zamboni et al. 2022

![sample_image](public/Zamboni_et_al._2021.png)

**Link:** [https://github.com/pedrozamboni/individual_urban_tree_crown_detection](https://github.com/pedrozamboni/individual_urban_tree_crown_detection)

**Location:** Mato Grosso do Sul, Brazil

# Points

## Amirkolaee et al. 2023

![sample_image](public/Amirkolaee_et_al._2023.png)

**Citation:** Amirkolaee, Hamed Amini, Miaojing Shi, and Mark Mulligan.  
*TreeFormer: a Semi-Supervised Transformer-based Framework for Tree Counting from a Single High Resolution Image*.  
IEEE Transactions on Geoscience and Remote Sensing (2023)

**Link:** [https://github.com/HAAClassic/TreeFormer](https://github.com/HAAClassic/TreeFormer)

**Location:** London, England

**GSD** 0.2m 

**Citation:** Lefebvre, I., Laliberté, E. (2024). UAV LiDAR, UAV Imagery, Tree Segmentations and Ground Mesurements for Estimating Tree Biomass in Canadian (Quebec) Plantations. Federated Research Data Repository. https://doi.org/10.20383/103.0979

## Ventura et al. 2022

![sample_image](public/Ventura_et_al._2022.png)

**Citation:** J. Ventura, C. Pawlak, M. Honsberger, C. Gonsalves, J. Rice, N.L.R. Love, S. Han, V. Nguyen, K. Sugano, J. Doremus, G.A. Fricker, J. Yost, and M. Ritter.  
*Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery*.  
International Journal of Applied Earth Observation and Geoinformation, 130, 103848 (2024)

**Link:** [https://github.com/jonathanventura/urban-tree-detection-data](https://github.com/jonathanventura/urban-tree-detection-data)

**Location:** Southern California, United States

**GSD:** 0.6m

## National Ecological Observatory Network

![sample_image](public/NEON_points.png)

**Location:** Multiple sites across the United States, see [NEON Field Sites](https://www.neonscience.org/field-sites/explore-field-sites)

**Link:** [https://data.neonscience.org/data-products/DP1.10098.001](https://data.neonscience.org/data-products/DP1.10098.001)

**GSD**: 0.1m 

## Auto-arborist

The auto-arborist dataset is a compilation of street-level surveys performed by local cities. Street trees were labeled with points. Annotations are limited to street trees, and do not include non-street trees. MillionTrees datasets are associated with state orthophoto programs that vary in ground resolution, from 20cm to 60cm. 

https://google.github.io/auto-arborist/

**Citation**: 
Sara Beery, Guanhang Wu, Trevor Edwards, Filip Pavetic, Bo Majewski, Shreyasee Mukherjee, Stanley Chan, John Morgan, Vivek Rathod, Jonathan Huang; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 21294-21307

**GSD**: 0.2m - 0.6m

# Polygons

## Araujo et al. 2020

![sample_image](public/Araujo_et_al._2020.png)

**Link:** [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243079][https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0243079]

**Location:** Manuas, Brazil

## Ball et al. 2023

**Link:** [https://zenodo.org/records/8136161](https://zenodo.org/records/8136161)

**Location:** Danum, Malaysia

## Bolhman 2008

![sample_image](public/Bohlman_et_al._2008.png)

Unpublished data from Barro Colorado Island, field verified after photo-interpretation.

Both crowns and points are available.

**Location**: BCI, Panama

## Cloutier et al. 2023

### Source Name: "Cloutier et al. 2023"

![sample_image](public/Cloutier_et_al._2023.png)

**Link:** [https://zenodo.org/records/8148479](https://zenodo.org/records/8148479)

**Location:** Quebec, Canada

## Feng et al. 2025

## Source Name: 'Feng et al. 2025' 

![sample_image](public/Feng_et_al._2025.png)

**Link:** [Zenodo](https://zenodo.org/records/14525290)
**Location:** Simulation

*Note this dataset is simulated and therefore somewhat more controvertial in its inclusion. We believe the size, diversity and realism of the images merit inclusion.*

## Firoze et al. 2023

### Source Name: "Firoze et al. 2023"

![sample_image](public/Firoze_et_al._2023.png)

**Link:** [https://openaccess.thecvf.com/content/CVPR2023/papers/Firoze_Tree_Instance_Segmentation_With_Temporal_Contour_Graph_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Firoze_Tree_Instance_Segmentation_With_Temporal_Contour_Graph_CVPR_2023_paper.pdf)

**Location:** Indiana, United States

## Hickman et al. 2021

**Link:** [https://zenodo.org/records/5515408](https://zenodo.org/records/5515408)

**Location:** Sabah, Malaysia

## Jansen et al. 2022

![sample_image](public/Jansen_et_al._2023.png)

**Link:** [https://zenodo.org/records/7094916](https://zenodo.org/records/7094916)

**Location:** Northern Australia

## JustDiggit

### Source Name: "Justdiggit 2023"

![sample_image](public/Justdiggit_2023.png)

**Link:** [JustDigIt](https://justdiggit.org/news/machine-learning-model-to-track-trees/)

**Location:** Tanzania

Citation status uncertain, contact Tjomme Dooper fruit punch AI.

## Li et al. 2023

![sample_image](public/Li_et_al._2023.png)

**link** [Manuscript](https://academic.oup.com/pnasnexus/article/2/4/pgad076/7073732?login=false) [github](https://github.com/sizhuoli/TreeCountSegHeight?tab=readme-ov-file)

**Location**: Denmark

## Lefebvre et al. 2024

### Source Name: "Lefebvre et al. 2024"

![sample_image](public/Lefebvre_et_al._2024.png)

[Dataset Link](https://www.frdr-dfdr.ca/repo/dataset/9f10a155-c89f-43ee-9864-da28ca436af6)

## Lucas et al. 2024

### Source Name: "Lucas et al. 2024"

![sample_image](public/Lucas_et_al._2024.png)

**Link:** [Manuscript](https://www.nw-fva.de/fileadmin/nwfva/publikationen/pdf/lucas_2024_individual_tree_detection_and_crown_delineation_in_the_harz.pdf) 

**Citation** Lucas, Moritz, et al. "Individual tree detection and crown delineation in the Harz National Park from 2009 to 2022 using mask R–CNN and aerial imagery." ISPRS Open Journal of Photogrammetry and Remote Sensing 13 (2024): 100071.

## Kattenborn et al. 2023

![sample_image](public/Kattenborn_et_al._2023.png)

**Link:** [https://www.mdpi.com/1424-8220/21/5/1617](https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.146)

**Location:** New Zealand

## Miranda et al. 2024

![sample_image](public/Alejandro_Miranda.png)

**Link:** [Courtesy of Alejandro Miranda](http://www.lepfor.ufro.cl/)

## Safonova et al. 2021

![sample_image](public/Safonova_et_al._2021.png)

**Link:** [https://www.mdpi.com/1424-8220/21/5/1617](https://www.mdpi.com/1424-8220/21/5/1617)

**Location:** Spain

## Takeshige et al. 2025

### Source Name: "Takeshige et al. 2025"

![sample_image](public/Takeshige_et_al._2025.png)

**Link:** https://esj-journals.onlinelibrary.wiley.com/doi/10.1111/1440-1703.12555

**Location:** Japan

## Troles et al. 2024

### Source Name: "Troles et al. 2024"

![sample_image](public/Troles_et_al._2024.png)

**Location:** Bamberg, Germany

## Wagner et al. 2023

![sample_image](public/Wagner_et_al._2023.png)

**Link:** [https://www.mdpi.com/2504-446X/7/3/155](https://www.mdpi.com/2504-446X/7/3/155)  
[https://www.mdpi.com/2072-4292/16/11/1935](https://www.mdpi.com/2072-4292/16/11/1935)

**Location:** Australia

## Vasquez et al. 2023

![sample_image](public/Vasquez_et_al._2023_-_training.png)
![sample_image](public/Vasquez_et_al._2023.png)

**Link:** [Figshare](https://smithsonian.figshare.com/articles/dataset/Barro_Colorado_Island_50-ha_plot_crown_maps_manually_segmented_and_instance_segmented_/24784053?file=43684731)  

**Location:** Barro Colorado Island, Panama

There is also a training-only portion of this that was used in conjuction with a model to predict labels that were then verified.   

# Unsupervised

## Weinstein et al. 2018

Coregistered LIDAR and RGB were acquired over 27 sites in the National Ecological Observation Network, USA. These sites cover a range of forest habitats. An unsupervised LiDAR tree detection algorithm was used to predict tree locations. These locatons were draped over the RGB data to create a very large weakly supervised dataset. There is currently over 40 million tree locations from the original dataset and more can be generated with ongoing data collection.

**Citation**: Weinstein, B.G.; Marconi, S.; Bohlman, S.; Zare, A.; White, E. Individual Tree-Crown Detection in RGB Imagery Using Semi-Supervised Deep Learning Neural Networks. Remote Sens. 2019, 11, 1309. https://doi.org/10.3390/rs11111309

**Location**: Forest across the United States (NEON)

## Open Forest Observatory

High resolution drone imagery used to create photogrametry-derived predictions of tree crowns.

https://openforestobservatory.org/

https://besjournals.onlinelibrary.wiley.com/doi/10.1111/2041-210X.13860

### Source Name: "Young et al. 2025 unsupervised"

***Location*** Forest across the United States