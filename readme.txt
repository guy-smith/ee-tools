EARTH ENGINE SNWA SPRING VALLEY README

RUNNING THE TOOLS
To generate the summary excel file, run the ee_summary_tables.py script
To generate the summary figues, run the ee_summary_figures.py script
When executing both of these scripts you pass an input or control filepath (*.ini) to the script using the "-i" or "--ini" command line argument.
    > python ee_summary_figures.py -i multipart\snwa_summary.ini
    
The "--debug" argument can also be passed to the scripts to have more additional details printed in the console.
    python ee_summary_figures.py -i multipart\snwa_summary.ini --debug

    
SUMMARY OPTIONS
To control which images and zones are included in the tables and summaryies, modify the summary control file ("snwa_summary.ini").
    The other control files in the folders (snwa_zs_north.ini and snwa_zs_south.ini) were used for generating the data and should not be used.
    
The default date range was all images from May-August (start_month=5 and end_month=9).
    To change this to April-Septempber, set start_month=4 and end_month=10.
The dates can also be controlled by day of year (DOY) using the start_doy and end_doy parameters.
    Simply remove the "#" from in from of the parameter 
    The "#" is a character that tells the code to ignore this line of the control file
    
The data can also be filtered by:
    Landsat type (set the landsatX_flags respectively)
    Landsat path/row
        Listed as comma separated values ("42, 43")
    A list of specific shapefile FIDs to "keep" or "skip"
        These can be listed as comma separated values ("1, 2, 3") or as a range ("1-10")
        
To control which months are included in the GRIDMET aggregation, set the gridmet_start_month and gridmet_end_month parameters
    If a gridmet_start_month value is set as 10, 11, or 12, these months will be included in the next calendar year.
    The range of the months will not exceed 12 months.
       If the start is 10 and end is 11, the range will only be 10-11
    To compute the water year sums, set the start to 10 and the end to 9

The summary tables script will include all available Landsat bands/products in the output excel files    
When generating summary figures, the Landsat bands/products can be controlled by setting the "figure_landsat_bands" and "complimentary_landsat_bands" parameters.
    The available options for figure_landsat_bands are: albedo_sur, ts, ndwi_sur, ndvi_sur, evi_sur
    The Ts band is not supported for the complimentary figures but it could probably be added.
    refl_sur, tasseled_cap, and fmask are not currently supported
    
    
FOLDERS
"images" folder
Landsat images for 2011-2016 and GRIDMET ETo and PPT images for 1984-2016 were downloaded for the Spring Valley hydrographic area.
Landsat images include the following products: NDVI, NDWI, EVI, surface temperature (Ts), Tasseled Cap, Albedo

"multipart" folder
Zonal statistics computed for original multipart polygons (extracted from geodatabase to shapefile)

"singlepart" folder
Zonal statistics compute for each separate polygon (split multipart polygons to singlepart using ArcGIS tool)
