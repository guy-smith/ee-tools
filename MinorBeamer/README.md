# Beamer ET*/ETg

**Note, the only difference between the image download INI files is the output workspace.**

## Beamer v0

Old style Beamer zonal stats with all output values stored in a single/common CSV file.
EDITS***:
New coefficients for the ET*-EVI relationship have been derived using the Landsat Collection 1 Surface Reflectance products and new ET* values for all Eddy Covariance/Bowen Ratio flux tower measurements (40 original site-years and more recent 16 site-years).

## ETg Zonal Stats

#### ee_beamer_zonal_stats.py

This script will compute Beamer ET*/ETg specific zonal statistics. 
EDITS***:
lines 722 through 772 contain new coefficients of the MinorBeamer ET*-VI predictive equation, as well as the uci, lci, lpi, and upi coeffs.

## ETg Images

There are three scripts for downloading ETg images.  The scripts will attempt to download separate images for each feature in the zones shapefile, so if you want a single image for all zones, the features must be dissolved/merged before running the script.

#### ee_beamer_image_download.py

This script will download each ETg image (based on the start/end date in the INI file) from Earth Engine.  The script will then compute the mean annual ETg for each year (water year) and the median and mean ETg for all years using the local images.

#### ee_beamer_image_annual_mean.py

This script will download the mean annual ETg for each year (water year) from Earth Engine.  The script will then compute the median and mean ETg for all years from the annuals using the local annual images.

#### ee_beamer_image_composite.py

This script will download the median and mean ETg for all years (water years) directly from Earth Engine.

## ETg Summary Tools

#### ee_beamer_summary_figures.py

This script will generate annual summary figures.

#### ee_beamer_summary_timeseries.py

This script will generate interactive bokeh timeseries plots of the zonal stats.

#### ee_beamer_summary_tables.py

This script will generate annual summary tables.

