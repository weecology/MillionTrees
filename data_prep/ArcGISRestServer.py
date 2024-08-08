# Download from ArcGIS Rest Service
import requests
import os
import sys
import time

#"usda":"https://gis.apfo.usda.gov/arcgis/rest/services/NAIP/USDA_CONUS_PRIME/ImageServer"

sites = {
    "calgary": "https://gis.calgary.ca/arcgis/rest/services/pub_Orthophotos/CurrentOrthophoto/ImageServer/exportImage?bbox=-7403.92%2C5641545.49%2C-7236.74%2C5641645.36&bboxSR=&size=&imageSR=&time=&format=tif&pixelType=U8&noData=&noDataInterpretation=esriNoDataMatchAny&interpolation=+RSP_BilinearInterpolation&compression=&compressionQuality=&bandIds=&sliceId=&mosaicRule=&renderingRule=&adjustAspectRatio=true&validateExtent=false&lercVersion=1&compressionTolerance=&f=html",
    "charlottesville":"https://gismaps.vdem.virginia.gov/arcgis/rest/services/VBMP_Imagery/MostRecentImagery_WGS_Tile_Index/MapServer",
    "bloomington":"https://imageserver.gisdata.mn.gov/cgi-bin/wms?",
    "new_york":"https://orthos.its.ny.gov/arcgis/rest/services/2018_4Band/ImageServer",
    "Washington_DC":"https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2023/ImageServer",
    "Edmonton":"https://gis.edmonton.ca/site1/rest/services/Imagery_Public/2019_RGB_Pictometry/ImageServer", # Also downloadable latest from https://drive.google.com/drive/folders/1c8rapuHkDuec_HzQoK27Bl3EIKovtDut
    "Pittsurgh":"https://imagery.pasda.psu.edu/arcgis/services/pasda/AlleghenyCountyImagery2017/MapServer/WMSServer?SERVICE=WMS&request=getcapabilities",#3 inch but leaf off
    "Sioux_Falls":"https://siouxfalls-my.sharepoint.com/personal/cityofsfgis_siouxfalls_gov/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fcityofsfgis%5Fsiouxfalls%5Fgov%2FDocuments%2FOpen%20Data%2FImagery%2F2023%2Fthree%2Dinch", # 3 inch but leaf off
    "Vancouver":"https://opendata.vancouver.ca/explore/dataset/orthophoto-imagery-2015/table/?location=12,49.25627,-123.1388",
    "Tempe":"https://data.tempe.gov/maps/cea011e890c847d297678a3176869bf3/explore?location=33.395241%2C-111.926973%2C19.00", # Can't download
    "Massachusetts":"https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/orthos2021/MapServer",
    "Hennepin":"https://gis.hennepin.us/arcgis/rest/services/Imagery/UTM_Aerial_2022/MapServer",
    }

# Oregon server https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2022
# Denver is alittle questionable, example tile. https://drapparchive.s3.amazonaws.com/2016/N1E192b.tif
# Download the file
def download_file(url, filename):
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

for site, html_request in sites.items():
    download_file(html_request, "{}.tif".format(site))