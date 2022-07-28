# ee-tools

Earth Engine (EE) Zonal Stats and Image Download Tools

The ee-tools can be separated into two main components:
+ Download time series of zonal statistics
+ Download daily Landsat and GRIDMET PPT/ETo imagery


## Study Area Zones

In order to compute zonal statistics, the user must provide a shapefile of the zones they wish to analyze.  Zonal statistics for each feature in the shapefile will be computed.  Images can be downloaded for each zone separately (set IMAGES parameter merge_geometries=False) or as a single image that includes all zones (merge_geometries=True).

The user is strongly encouraged to "rasterize" the study area zones to the UTM Zones of the intersecting Landsat path/rows using the provided tool (miscellaneous/polygon_rasterize.py).  This script will adjust the zone geometries to follow the Landsat pixels.  For example, a field in Mason Valley, NV is in the overlap of two Landsat path/rows with different UTM Zones (42/33 - WGS84 Zone 11N / EPSG:32611 and 43/44 - Zone 10N / EPSG:32610), so separate rasterized shapefiles should be generated for each UTM zone.

To rasterize the example shapefile to EPSG 32610 and EPSG 32611, execute the following:
```
> python ..\miscellaneous\polygon_rasterize.py example.shp example_wgs84z10.shp --epsg 32610 -o
> python ..\miscellaneous\polygon_rasterize.py example.shp example_wgs84z11.shp --epsg 32611 -o
```

#### Zone Field

The user must indicate which field in the shapefile to use for setting the "Zone ID".  The field must be an integer or string type and the values must be unique for each feature/zone.  A good default is use the "FID" field since this is guaranteed to be unique and makes it easy to join the output tables to the shapefile.

#### Spatial Reference / Projection

Currently, the output spatial reference set in the INI file (EXPORT parameter "output_proj") must match exactly with the spatial reference of the zones shapefile.  The code should prompt you if they do not match, in which case you should reproject the zones shapefile to the output spatial reference (see [Study Area Zones](Study Area Zones)).  Eventually the code will be able to project the zones geometries to the output projection automatically.


## Zonal Statistics

To initiate Earth Engine zonal statistics calculations, run the following command:
```
> python ee_zonal_stats_by_zone.py -i example\example_zonal_stats.ini
```

#### Output

Separate CSV files will be written for each feature in the shapefile to the folder specified in the "output_workspace" parameter of the INI file.

Along with the Landsat zonal statistics values, the following fields will also be written to the CSV file:
* PIXEL_TOTAL - Number of pixels that could nominally be in the zone.
* PIXEL_COUNT - Number of pixels with data used in the computation of mean NDVI, Ts, etc.  PIXEL_COUNT should always be <= PIXEL_TOTAL.  PIXEL_COUNT will be lower than PIXEL_TOTAL for zones that are near the edge of the image or cross the scan-line corrector gaps in Landsat 7 images.  Zones that are fully contained within cloud free Landsat 5 and 8 images can have PIXEL_COUNTS equal to PIXEL_TOTAL.
* FMASK_TOTAL - Number of pixels with an FMASK value.  FMASK_TOTAL should be equal to PIXEL_COUNT, but may be slightly different for LE7 SCL-off images.
* FMASK_COUNT - Number of pixels with FMASK values of 2, 3, or 4 (shadow, snow, and cloud).  FMASK_COUNT should always be <= FMASK_TOTAL.  Cloudy scenes will have high FMASK_COUNTs relative to FMASK_TOTAL.
* FMASK_PCT - Percentage of available pixels that are cloudy (FMASK_COUNT / FMASK_TOTAL)
* QA - QA/QC value (higher values are more likely to be cloudy or bad data).  This field is computed by the summary_qaqc.py script in the "summary-tools" repository.


## Image Download

To download Landsat images, execute the following:
```
> python ee_landsat_image_download.py -i example\example_images.ini
```

To download GRIDMET ETo/PPT images, execute the following:
```
> python ee_gridmet_image_download.py -i example\example_images.ini
```

The download scripts must be run twice in order to first export the TIF files to your Google drive and then copy them to the output workspace.

## Landsat Thumbnail Download

To download Landsat thumbnail images for each zone, execute the following:
```
> python ee_landsat_thumbnail_download.py -i example\example_summary.ini
```

Currently you must use the "summary" or "zonal_stats" INI file to set the output workspace.
The Landsat thumbnail script must also be run after zonal statistics have been computed, since it reads the landsat_daily.csv to determine which images to download.


## INI Files

All of the scripts are controlled using INI files.  The INI file is structured into sections (defined by square brackets, i.e. [INPUTS]) and key/value pairs separated by an equals sign (i.e. "start_year = 1985").  Additional details on the INI structure can be found in the [Python configparser module documentation](https://docs.python.org/3/library/configparser.html#supported-ini-file-structure).  Example INI files are provided in the example folder.

To set the input file, use the "-i" or "--ini" argument.  The INI file path can be absolute or relative to the current working directory.
```
> python ee_zonal_stats_by_zone.py -i example\example_zonal_stats.ini
```

#### Sections

Each of the scripts reads a different combination of INI sections.  There are seven sections currently used in the scripts:
+ INPUTS - Used by all of the ee-tools
+ EXPORT - Export specific parameters and is read by the zonal statistics and image download scripts.
+ ZONAL_STATS - Zonal stats specific parameters.
+ SPATIAL - Spatial reference (projection, snap point, cellsize) parameters.
+ IMAGES - Image download specific parameters.
+ SUMMARY - Summary specific parameters and is read by the summary figures and summary tables scripts.
+ FIGURES - Summary figure specific parameters.
+ TABLES - Summary table specific parameters.


## Requirements

#### Python

For information on installing Python and the required external modules, please see the [Python README](PYTHON.md).

#### Earth Engine

To run the ee-tools you must have an Earth Engine account.  If you do not have an  account, please go the Earth Engine signup page: https://signup.earthengine.google.com

#### Google Drive

*Note, this is only needed if using the GDRIVE export destination and is not a requirement*
To run the zonal stats and image download scripts, you must have Google Drive installed on your computer.  The scripts first initiate Earth Engine export tasks that will write to your Google Drive, and then copy the files from Google Drive to the output workspace set in the INI file.


## Scripts

#### Command Prompt / Terminal

All of the python scripts should be run from the terminal (mac/linux) or command prompt (windows).

In some cases the scripts can also be run by double clicking directly on the script.  The script will open a GUI asking you select an INI file.  Be advised, if you have multiple versions of Python installed (for example if you have ArcGIS and you install Anaconda), this may try to use a different different version of Python.

#### Help
To see what arguments are available for a script, and their default values, pass the "--help" or "-h" argument to the script.
```
> python ee_zonal_stats_by_zone.py -h
usage: ee_zonal_stats_by_zone.py [-h] [-i PATH] [-d] [-o]

Earth Engine Zonal Statistics

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --ini FILE   Input file (default: None)
  -d, --debug           Debug level logging (default: 20)
  -o, --overwrite       Force overwrite of existing files (default: False)
```

#### Debug

Debug mode can be enabled for all scripts by passing the "--debug" or "-d" argument to the script:
```
> python ee_shapefile_zonal_stats_export.py --debug
```

#### Overwrite

Overwrite mode can be enabled for many scripts by passing the "--ovewrite" or "-o" argument to the script:
```
> python ee_shapefile_zonal_stats_export.py --overwrite
```
