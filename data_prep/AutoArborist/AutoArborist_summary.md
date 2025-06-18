# AutoArborist Dataset Summary

Automated overhead imagery acquisition and annotation for tree locations.

## Overview

- **Total cities processed**: 1
- **Successful cities**: 1
- **Total tree annotations**: 4

## Cities with Successful Imagery

- **calgary**: 4 trees

## Failed Cities


## Dataset Format

The AutoArborist dataset follows the MillionTrees format with point annotations:

- `image_path`: Relative path to the image file
- `x`, `y`: Pixel coordinates of tree locations
- `label`: Always 'Tree'
- `source`: 'AutoArborist'
- `score`: 1.0 (ground truth)
- `genus`: Tree genus from original data
- `taxonomy_id`: Taxonomy ID from original data

## Imagery Sources

- **Calgary**: https://gis.calgary.ca/arcgis/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/exportImage
- **Edmonton**: https://gis.edmonton.ca/site1/rest/services/Imagery_Public/2019_RGB_Pictometry/ImageServer/exportImage
- **Vancouver**: https://opendata.vancouver.ca/explore/dataset/orthophoto-imagery-2015/api
- **New_York**: https://orthos.its.ny.gov/arcgis/rest/services/2018_4Band/ImageServer/exportImage
- **Washington_Dc**: https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2023/ImageServer/exportImage
