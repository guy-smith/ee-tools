#--------------------------------
# Name:         ee_shapefile_zonal_stats_offline.py
# Purpose:      Compute offline zonal stats for shapefiles
# Updated:      2016-10-17
# Python:       2.7
#--------------------------------

import argparse
# from collections import defaultdict
import ConfigParser
import datetime
# import json
import logging
import os
import re
import sys

from dateutil.relativedelta import relativedelta
import numpy as np
from osgeo import gdal, ogr, osr
import pandas as pd

import common
import gdal_common as gdc


# Add option to apply generic mask
# Something called .mask.tif ?

def zonal_stats(ini_path=None, overwrite_flag=False):
    """Offline Zonal Stats

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nCompute Offline Zonal Stats')

    landsat_flag = True
    gridmet_flag = True
    pdsi_flag = False

    landsat_images_folder = 'landsat'
    landsat_tables_folder = 'landsat_tables'
    gridmet_images_folder = 'gridmet_monthly'

    # Regular expression to pull out Landsat scene_id
    landsat_image_re = re.compile('^\d{8}_\d{3}_\w+.\w+.tif$')
    gridmet_image_re = re.compile('^\d{6}_gridmet.(eto|ppt).tif$')

    # For now, hardcode snap, cellsize and spatial reference
    logging.info('\nHardcoding zone/output cellsize and snap')
    zone_cs = 30
    zone_x, zone_y = 15, 15
    logging.debug('  Snap: {} {}'.format(zone_x, zone_y))
    logging.debug('  Cellsize: {}'.format(zone_cs))

    logging.info('Hardcoding Landsat snap, cellsize and spatial reference')
    landsat_x, landsat_y = 15, 15
    landsat_cs = 30
    landsat_osr = gdc.epsg_osr(32611)
    logging.debug('  Snap: {} {}'.format(landsat_x, landsat_y))
    logging.debug('  Cellsize: {}'.format(landsat_cs))
    logging.debug('  OSR: {}'.format(landsat_osr))

    logging.info('Hardcoding GRIDMET snap, cellsize and spatial reference')
    gridmet_x, gridmet_y = -124.79299639209513, 49.41685579737572
    gridmet_cs = 0.041666001963701
    # gridmet_cs = [0.041666001963701, 0.041666001489718]
    # gridmet_x, gridmet_y = -124.79166666666666666667, 25.04166666666666666667
    # gridmet_cs = 1. / 24
    gridmet_osr = gdc.epsg_osr(4326)
    # gridmet_osr = gdc.epsg_osr(4269)
    logging.debug('  Snap: {} {}'.format(gridmet_x, gridmet_y))
    logging.debug('  Cellsize: {}'.format(gridmet_cs))
    logging.debug('  OSR: {}'.format(gridmet_osr))

    landsat_daily_fields = [
        'DATE', 'SCENE_ID', 'LANDSAT', 'PATH', 'ROW',
        'YEAR', 'MONTH', 'DAY', 'DOY',
        'PIXEL_COUNT', 'FMASK_COUNT', 'DATA_COUNT', 'CLOUD_SCORE',
        'TS', 'ALBEDO_SUR', 'NDVI_TOA', 'NDVI_SUR', 'EVI_SUR',
        'NDWI_GREEN_NIR_SUR', 'NDWI_GREEN_SWIR1_SUR', 'NDWI_NIR_SWIR1_SUR',
        # 'NDWI_GREEN_NIR_TOA', 'NDWI_GREEN_SWIR1_TOA', 'NDWI_NIR_SWIR1_TOA',
        # 'NDWI_SWIR1_GREEN_TOA', 'NDWI_SWIR1_GREEN_SUR',
        # 'NDWI_TOA', 'NDWI_SUR',
        'TC_BRIGHT', 'TC_GREEN', 'TC_WET']
    # gridmet_daily_fields = [
    #     'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY', 'WATER_YEAR', 'ETO', 'PPT']
    gridmet_monthly_fields = [
        'DATE', 'YEAR', 'MONTH', 'WATER_YEAR', 'ETO', 'PPT']
    pdsi_dekad_fields = [
        'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY', 'PDSI']

    landsat_int_fields = [
        'YEAR', 'MONTH', 'DAY', 'DOY',
        'PIXEL_COUNT', 'FMASK_COUNT', 'CLOUD_SCORE']
    gridmet_int_fields = ['YEAR', 'MONTH', 'WATER_YEAR']

    # To figure out which Landsat and path,
    # Compare date to reference dates and look for even multiples of 16
    ref_dates = {
        datetime.datetime(1985, 3, 31): ['LT5', '039'],
        datetime.datetime(1985, 4, 7): ['LT5', '040'],
        datetime.datetime(1999, 7, 4): ['LE7', '039'],
        datetime.datetime(1999, 7, 27): ['LE7', '040'],
        datetime.datetime(2013, 4, 13): ['LC8', '039'],
        datetime.datetime(2013, 4, 20): ['LC8', '040']
        # datetime.datetime(1984, , ): ['LT4', '039'],
        # datetime.datetime(1984, , ): ['LT4', '040'],
    }

    # Open config file
    config = ConfigParser.ConfigParser()
    try:
        config.readfp(open(ini_path))
    except:
        logging.error(('\nERROR: Input file could not be read, ' +
                       'is not an input file, or does not exist\n' +
                       'ERROR: ini_path = {}\n').format(ini_path))
        sys.exit()
    logging.debug('\nReading Input File')

    # Read in config file
    zone_input_ws = config.get('INPUTS', 'zone_input_ws')
    zone_filename = config.get('INPUTS', 'zone_filename')
    zone_field = config.get('INPUTS', 'zone_field')
    zone_path = os.path.join(zone_input_ws, zone_filename)

    landsat_daily_fields.insert(0, zone_field)
    # gridmet_daily_fields.insert(0, zone_field)
    gridmet_monthly_fields.insert(0, zone_field)
    pdsi_dekad_fields.insert(0, zone_field)

    images_ws = config.get('INPUTS', 'images_ws')

    # Build and check file paths
    if not os.path.isdir(zone_input_ws):
        logging.error(
            '\nERROR: The zone workspace does not exist, exiting\n  {}'.format(
                zone_input_ws))
        sys.exit()
    elif not os.path.isfile(zone_path):
        logging.error(
            '\nERROR: The zone shapefile does not exist, exiting\n  {}'.format(
                zone_path))
        sys.exit()
    elif not os.path.isdir(images_ws):
        logging.error(
            '\nERROR: The image workspace does not exist, exiting\n  {}'.format(
                images_ws))
        sys.exit()

    # Final output folder
    try:
        output_ws = config.get('INPUTS', 'output_ws')
        if not os.path.isdir(output_ws):
            os.makedirs(output_ws)
    except:
        output_ws = os.getcwd()
        logging.debug('  Defaulting output workspace to {}'.format(output_ws))

    # Start/end year
    try:
        start_year = int(config.get('INPUTS', 'start_year'))
    except:
        start_year = 1984
        logging.debug('  Defaulting start_year={}'.format(start_year))
    try:
        end_year = int(config.get('INPUTS', 'end_year'))
    except:
        end_year = datetime.datetime.today().year
        logging.debug('  Defaulting end year to {}'.format(end_year))
    if start_year and end_year and end_year < start_year:
        logging.error(
            '\nERROR: End year must be >= start year, exiting')
        sys.exit()
    default_end_year = datetime.datetime.today().year + 1
    if (start_year and start_year not in range(1984, default_end_year) or
        end_year and end_year not in range(1984, default_end_year)):
        logging.error(
            ('\nERROR: Year must be an integer from 1984-{}, ' +
             'exiting').format(default_end_year - 1))
        sys.exit()

    # Start/end month
    try:
        start_month = int(config.get('INPUTS', 'start_month'))
    except:
        start_month = None
        logging.debug('  Defaulting start_month=None')
    try:
        end_month = int(config.get('INPUTS', 'end_month'))
    except:
        end_month = None
        logging.debug('  Defaulting end_month=None')
    if start_month and start_month not in range(1, 13):
        logging.error(
            '\nERROR: Start month must be an integer from 1-12, exiting')
        sys.exit()
    elif end_month and end_month not in range(1, 13):
        logging.error(
            '\nERROR: End month must be an integer from 1-12, exiting')
        sys.exit()
    month_list = common.wrapped_range(start_month, end_month, 1, 12)

    # Start/end DOY
    try:
        start_doy = int(config.get('INPUTS', 'start_doy'))
    except:
        start_doy = None
        logging.debug('  Defaulting start_doy=None')
    try:
        end_doy = int(config.get('INPUTS', 'end_doy'))
    except:
        end_doy = None
        logging.debug('  Defaulting end_doy=None')
    if end_doy and end_doy > 273:
        logging.error(
            '\nERROR: End DOY must be in the same water year as start DOY, ' +
            'exiting')
        sys.exit()
    if start_doy and start_doy not in range(1, 367):
        logging.error(
            '\nERROR: Start DOY must be an integer from 1-366, exiting')
        sys.exit()
    elif end_doy and end_doy not in range(1, 367):
        logging.error(
            '\nERROR: End DOY must be an integer from 1-366, exiting')
        sys.exit()
    # if end_doy < start_doy:
    #     logging.error(
    #         '\nERROR: End DOY must be >= start DOY')
    #     sys.exit()
    doy_list = common.wrapped_range(start_doy, end_doy, 1, 366)

    # Control which Landsat images are used
    try:
        landsat5_flag = config.getboolean('INPUTS', 'landsat5_flag')
    except:
        landsat5_flag = False
        logging.debug('  Defaulting landsat5_flag=False')
    try:
        landsat4_flag = config.getboolean('INPUTS', 'landsat4_flag')
    except:
        landsat4_flag = False
        logging.debug('  Defaulting landsat4_flag=False')
    try:
        landsat7_flag = config.getboolean('INPUTS', 'landsat7_flag')
    except:
        landsat7_flag = False
        logging.debug('  Defaulting landsat7_flag=False')
    try:
        landsat8_flag = config.getboolean('INPUTS', 'landsat8_flag')
    except:
        landsat8_flag = False
        logging.debug('  Defaulting landsat8_flag=False')

    # Cloudmasking
    try:
        apply_mask_flag = config.getboolean('INPUTS', 'apply_mask_flag')
    except:
        apply_mask_flag = False
        logging.debug('  Defaulting apply_mask_flag=False')

    try:
        acca_flag = config.getboolean('INPUTS', 'acca_flag')
    except:
        acca_flag = False
    try:
        fmask_flag = config.getboolean('INPUTS', 'fmask_flag')
    except:
        fmask_flag = False

    # Intentionally don't apply scene_id skip/keep lists
    # Compute zonal stats for all available images
    # Filter by scene_id when making summary tables
    scene_id_keep_list = []
    scene_id_skip_list = []

    # # Only process specific Landsat scenes
    # try:
    #     scene_id_keep_path = config.get('INPUTS', 'scene_id_keep_path')
    #     with open(scene_id_keep_path) as input_f:
    #         scene_id_keep_list = input_f.readlines()
    #     scene_id_keep_list = [x.strip()[:16] for x in scene_id_keep_list]
    # except IOError:
    #     logging.error('\nFileIO Error: {}'.format(scene_id_keep_path))
    #     sys.exit()
    # except:
    #     scene_id_keep_list = []

    # # Skip specific landsat scenes
    # try:
    #     scene_id_skip_path = config.get('INPUTS', 'scene_id_skip_path')
    #     with open(scene_id_skip_path) as input_f:
    #         scene_id_skip_list = input_f.readlines()
    #     scene_id_skip_list = [x.strip()[:16] for x in scene_id_skip_list]
    # except IOError:
    #     logging.error('\nFileIO Error: {}'.format(scene_id_skip_path))
    #     sys.exit()
    # except:
    #     scene_id_skip_list = []

    # Only process certain Landsat path/rows
    try:
        path_keep_list = list(
            common.parse_int_set(config.get('INPUTS', 'path_keep_list')))
    except:
        path_keep_list = []
    # try:
    #     row_keep_list = list(
    #         common.parse_int_set(config.get('INPUTS', 'row_keep_list')))
    # except:
    #     row_keep_list = []

    # Skip or keep certain FID
    try:
        fid_skip_list = list(
            common.parse_int_set(config.get('INPUTS', 'fid_skip_list')))
    except:
        fid_skip_list = []
    try:
        fid_keep_list = list(
            common.parse_int_set(config.get('INPUTS', 'fid_keep_list')))
    except:
        fid_keep_list = []

    # For now, output projection must be manually set above to match zones
    zone_osr = gdc.feature_path_osr(zone_path)
    zone_proj = gdc.osr_proj(zone_osr)
    logging.info('\nThe zone shapefile must be in a projected coordinate system!')
    logging.info('  Proj4: {}'.format(zone_osr.ExportToProj4()))
    logging.info('{}'.format(zone_osr))


    # Read in zone shapefile
    logging.info('\nRasterizing Zone Shapefile')
    zone_name_dict = dict()
    zone_extent_dict = dict()
    zone_mask_dict = dict()

    # First get FIDs and extents
    zone_ds = ogr.Open(zone_path, 0)
    zone_lyr = zone_ds.GetLayer()
    zone_lyr.ResetReading()
    for zone_ftr in zone_lyr:
        zone_fid = zone_ftr.GetFID()
        if zone_field.upper() == 'FID':
            zone_name_dict[zone_fid] = str(zone_fid)
        else:
            zone_name_dict[zone_fid] = zone_ftr.GetField(zone_field)
        zone_extent = gdc.Extent(
            zone_ftr.GetGeometryRef().GetEnvelope()).ogrenv_swap()
        zone_extent.adjust_to_snap('EXPAND', zone_x, zone_y, zone_cs)
        zone_extent_dict[zone_fid] = list(zone_extent)

    # Rasterize each FID separately
    # The RasterizeLayer function wants a "layer"
    # There might be an easier way to select each feature as a layer
    for zone_fid, zone_extent in sorted(zone_extent_dict.items()):
        logging.debug('FID: {}'.format(zone_fid))
        logging.debug('  Name: {}'.format(zone_name_dict[zone_fid]))
        zone_ds = ogr.Open(zone_path, 0)
        zone_lyr = zone_ds.GetLayer()
        zone_lyr.ResetReading()
        zone_lyr.SetAttributeFilter("{0} = {1}".format('FID', zone_fid))

        zone_extent = gdc.Extent(zone_extent)
        zone_rows, zone_cols = zone_extent.shape(zone_cs)
        logging.debug('  Extent: {}'.format(str(zone_extent)))
        logging.debug('  Rows/Cols: {} {}'.format(zone_rows, zone_cols))

        # zones_lyr.SetAttributeFilter("{0} = {1}".format('FID', zone_fid))

        # Initialize the zone in memory raster
        mem_driver = gdal.GetDriverByName('MEM')
        zone_raster_ds = mem_driver.Create(
            '', zone_cols, zone_rows, 1, gdal.GDT_Byte)
        zone_raster_ds.SetProjection(zone_proj)
        zone_raster_ds.SetGeoTransform(
            gdc.extent_geo(zone_extent, cs=zone_cs))
        zone_band = zone_raster_ds.GetRasterBand(1)
        zone_band.SetNoDataValue(0)

        # Clear the raster before rasterizing
        zone_band.Fill(0)
        gdal.RasterizeLayer(zone_raster_ds, [1], zone_lyr)
        # zones_ftr_ds = None
        zone_array = gdc.raster_ds_to_array(
            zone_raster_ds, return_nodata=False)
        zone_mask = zone_array != 0
        logging.debug('  Pixel Count: {}'.format(np.sum(zone_mask)))
        # logging.debug('  Mask:\n{}'.format(zone_mask))
        # logging.debug('  Array:\n{}'.format(zone_array))
        zone_mask_dict[zone_fid] = zone_mask

        zone_raster_ds = None
        del zone_raster_ds, zone_array, zone_mask
    zone_ds = None
    del zone_ds, zone_lyr



    # Calculate zonal stats for each feature separately
    logging.info('')
    for fid, zone_str in sorted(zone_name_dict.items()):
        if fid_keep_list and fid not in fid_keep_list:
            continue
        elif fid_skip_list and fid in fid_skip_list:
            continue
        logging.info('ZONE: {} (FID: {})'.format(zone_str, fid))

        if not zone_field or zone_field.upper() == 'FID':
            zone_str = 'fid_' + zone_str
        else:
            zone_str = zone_str.lower().replace(' ', '_')

        zone_output_ws = os.path.join(output_ws, zone_str)
        if not os.path.isdir(zone_output_ws):
            os.makedirs(zone_output_ws)

        zone_extent = gdc.Extent(zone_extent_dict[fid])
        zone_mask = zone_mask_dict[fid]
        # logging.debug('  Extent: {}'.format(zone_extent))


        if landsat_flag:
            logging.info('  Landsat')

            landsat_output_ws = os.path.join(
                zone_output_ws, landsat_tables_folder)
            if not os.path.isdir(landsat_output_ws):
                os.makedirs(landsat_output_ws)
            logging.debug('  {}'.format(landsat_output_ws))

            # Project the zone extent to the image OSR
            clip_extent = gdc.project_extent(
                zone_extent, zone_osr, landsat_osr, zone_cs)
            # logging.debug('  Extent: {}'.format(clip_extent))
            clip_extent.adjust_to_snap('EXPAND', landsat_x, landsat_y, landsat_cs)
            logging.debug('  Extent: {}'.format(clip_extent))

            # Process date range by year
            for year in xrange(start_year, end_year + 1):
                images_year_ws = os.path.join(
                    images_ws, landsat_images_folder, str(year))
                if not os.path.isdir(images_year_ws):
                    logging.debug(
                        '  Landsat year folder doesn\'t exist, skipping\n    {}'.format(
                            images_year_ws))
                    continue
                else:
                    logging.info('  Year: {}'.format(year))

                # Create an empty dataframe
                output_path = os.path.join(
                    landsat_output_ws, '{}_landsat_{}.csv'.format(zone_str, year))
                if os.path.isfile(output_path):
                    if overwrite_flag:
                        logging.debug(
                            '  Output CSV already exists, removing\n    {}'.format(
                                output_path))
                        os.remove(output_path)
                    else:
                        logging.debug(
                            '  Output CSV already exists, skipping\n    {}'.format(
                                output_path))
                        continue
                output_df = pd.DataFrame(columns=landsat_daily_fields)
                output_df[landsat_int_fields] = output_df[
                    landsat_int_fields].astype(int)

                # Get list of all images
                year_image_list = [
                    image for image in os.listdir(images_year_ws)
                    if landsat_image_re.match(image)]
                # Get list of all unique dates (multiple images per date)
                year_dt_list = sorted(set([
                    datetime.datetime.strptime(image[:8], '%Y%m%d')
                    for image in year_image_list]))
                # Filter date lists if necessary
                if month_list:
                    year_dt_list = [
                        image_dt for image_dt in year_dt_list
                        if image_dt.month in month_list]
                if doy_list:
                    year_dt_list = [
                        image_dt for image_dt in year_dt_list
                        if int(image_dt.strftime('%j')) in doy_list]

                output_list = []
                for image_dt in year_dt_list:
                    image_str = image_dt.date().isoformat()
                    logging.debug('{}'.format(image_dt.date()))

                    # Get the list of available images
                    image_list = [
                        image for image in year_image_list
                        if image_dt.strftime('%Y%m%d') in image]
                    # This conditional is probably impossible
                    if not image_list:
                        logging.debug('    No images, skipping date')
                        continue

                    # Use date offsets to determine the Landsat and Path
                    ref_match = [
                        lp for ref_dt, lp in ref_dates.items()
                        if (((ref_dt - image_dt).days % 16 == 0) and
                            ((lp[0].upper() == 'LT5' and image_dt.year < 2012) or
                             (lp[0].upper() == 'LC8' and image_dt.year > 2012) or
                             (lp[0].upper() == 'LE7')))]
                    if ref_match:
                        landsat, path = ref_match[0]
                    else:
                        landsat, path = 'XXX', '000'
                    # Get Landsat type from first image in list
                    # image_dict['LANDSAT'] = image_list[0].split('.')[0].split('_')[2]
                    image_name_fmt = '{}_{}.{}.tif'.format(
                        image_dt.strftime('%Y%m%d_%j'), landsat.lower(), '{}')

                    if not landsat4_flag and landsat.upper() == 'LT4':
                        logging.debug('    Landsat 4, skipping image')
                        continue
                    elif not landsat5_flag and landsat.upper() == 'LT5':
                        logging.debug('    Landsat 5, skipping image')
                        continue
                    elif not landsat7_flag and landsat.upper() == 'LE7':
                        logging.debug('    Landsat 7, skipping image')
                        continue
                    elif not landsat8_flag and landsat.upper() == 'LC8':
                        logging.debug('    Landsat 8, skipping image')
                        continue

                    # Load the "mask" image first if it is available
                    # The zone_mask could be applied to the mask_array here
                    #   or below where it is used to select from the image_array
                    mask_name = image_name_fmt.format('mask')
                    mask_path = os.path.join(images_year_ws, mask_name)
                    if apply_mask_flag and mask_name in image_list:
                        logging.info('    Applying mask raster: {}'.format(
                            mask_path))
                        mask_input_array, mask_nodata = gdc.raster_to_array(
                            mask_path, band=1, mask_extent=clip_extent,
                            fill_value=None, return_nodata=True)
                        mask_array = gdc.project_array(
                            mask_input_array, gdal.GRA_NearestNeighbour,
                            landsat_osr, landsat_cs, clip_extent,
                            zone_osr, zone_cs, zone_extent,
                            output_nodata=None)
                        # Assume 0 and nodata indicate unmasked pixels
                        # All other pixels are "masked"
                        mask_array = (mask_array == 0) | (mask_array == mask_nodata)
                        # Assume 0 and nodata indicate masked pixels
                        # mask_array = (mask_array != 0) & (mask_array != mask_nodata)
                        if not np.any(mask_array):
                            logging.info('    No unmasked values')
                    else:
                        mask_array = np.ones(zone_mask.shape, dtype=np.bool)

                    # Save date specific properties
                    image_dict = dict()

                    # Get Fmask and Cloud score separately from other bands
                    # FMask
                    image_name = image_name_fmt.format('fmask')
                    image_path = os.path.join(images_year_ws, image_name)
                    if not os.path.isfile(image_path):
                        logging.error(
                            '  Image {} does not exist, skipping date'.format(
                                image_name))
                        continue
                    image_input_array, image_nodata = gdc.raster_to_array(
                        image_path, band=1, mask_extent=clip_extent,
                        fill_value=None, return_nodata=True)
                    fmask_array = gdc.project_array(
                        image_input_array, gdal.GRA_NearestNeighbour,
                        landsat_osr, landsat_cs, clip_extent,
                        zone_osr, zone_cs, zone_extent,
                        output_nodata=None)
                    fmask_mask = np.copy(zone_mask) & mask_array
                    if fmask_array.dtype in [np.float32, np.float64]:
                        fmask_mask &= np.isfinite(fmask_array)
                    else:
                        fmask_mask &= fmask_array != image_nodata
                    if not np.any(fmask_mask):
                        logging.debug('    Empty Fmask array, skipping')
                        continue
                    # Convert Fmask array into a mask (1 is cloudy, 0 is clear)
                    fmask_array = (fmask_array > 1.5) & (fmask_array < 4.5)
                    image_dict['FMASK_COUNT'] = int(np.sum(fmask_array[fmask_mask]))
                    image_dict['PIXEL_COUNT'] = int(np.sum(fmask_mask))
                    # image_dict['PIXEL_COUNT'] = int(np.sum(fmask_mask))
                    image_dict['MASK_COUNT'] = int(np.sum(mask_array))

                    # Cloud Score
                    image_name = image_name_fmt.format('cloud_score')
                    image_path = os.path.join(images_year_ws, image_name)
                    image_input_array, image_nodata = gdc.raster_to_array(
                        image_path, band=1, mask_extent=clip_extent,
                        fill_value=None, return_nodata=True)
                    cloud_array = gdc.project_array(
                        image_input_array, gdal.GRA_NearestNeighbour,
                        landsat_osr, landsat_cs, clip_extent,
                        zone_osr, zone_cs, zone_extent,
                        output_nodata=None)
                    cloud_mask = np.copy(zone_mask) & mask_array
                    if cloud_array.dtype in [np.float32, np.float64]:
                        cloud_mask &= np.isfinite(cloud_array)
                    else:
                        cloud_mask &= cloud_array != image_nodata
                    if not np.any(cloud_mask):
                        logging.debug('    Empty Cloud Score array, skipping')
                        continue
                    image_dict['CLOUD_SCORE'] = float(np.mean(cloud_array[cloud_mask]))


                    # Workflow
                    zs_list = [
                        ['ts', 1, 'TS'],
                        ['albedo_sur', 1, 'ALBEDO_SUR'],
                        ['ndvi_toa', 1, 'NDVI_TOA'],
                        ['ndvi_sur', 1, 'NDVI_SUR'],
                        ['evi_sur', 1, 'EVI_SUR'],
                        ['ndwi_green_nir_sur', 1, 'NDWI_GREEN_NIR_SUR'],
                        ['ndwi_green_swir1_sur', 1, 'NDWI_GREEN_SWIR1_SUR'],
                        ['ndwi_nir_swir1_sur', 1, 'NDWI_NIR_SWIR1_SUR'],
                        ['tasseled_cap', 1, 'TC_BRIGHT'],
                        ['tasseled_cap', 2, 'TC_GREEN'],
                        ['tasseled_cap', 3, 'TC_WET']
                    ]
                    for band_name, band_num, field in zs_list:
                        image_name = image_name_fmt.format(band_name)
                        logging.debug('  {} {}'.format(image_name, field))
                        if image_name not in image_list:
                            logging.debug('    Image doesn\'t exist, skipping')
                            continue
                        image_path = os.path.join(images_year_ws, image_name)
                        # logging.debug('  {}'.format(image_path))

                        image_input_array, image_nodata = gdc.raster_to_array(
                            image_path, band=band_num, mask_extent=clip_extent,
                            fill_value=None, return_nodata=True)

                        # GRA_NearestNeighbour, GRA_Bilinear, GRA_Cubic,
                        #   GRA_CubicSpline
                        image_array = gdc.project_array(
                            image_input_array, gdal.GRA_NearestNeighbour,
                            landsat_osr, landsat_cs, clip_extent,
                            zone_osr, zone_cs, zone_extent,
                            output_nodata=None)
                        image_mask = np.copy(zone_mask) & mask_array
                        if image_array.dtype in [np.float32, np.float64]:
                            image_mask &= np.isfinite(image_array)
                        else:
                            image_mask &= image_array != image_nodata
                        del image_input_array

                        if fmask_flag:
                            # Fmask array was converted into a mask
                            # 1 for cloud, 0 for clear
                            image_mask &= (fmask_array == 0)
                        if acca_flag:
                            image_mask &= (cloud_array < 50)

                        # Skip fully masked zones
                        # This would not work for FMASK and CLOUD_SCORE if we
                        #   weren't using nearest neighbor for resampling
                        if not np.any(image_mask):
                            logging.debug('    Empty array, skipping')
                            continue

                        image_dict[field] = float(np.mean(
                            image_array[image_mask]))

                        # Should check "first" image instead of Ts specifically
                        if band_name == 'ts':
                            image_dict['DATA_COUNT'] = int(np.sum(image_mask))

                        del image_array, image_mask

                    if not image_dict:
                        logging.debug(
                            '    {} - no image data in zone, skipping'.format(
                                image_str))
                        continue

                    # Save date specific properties
                    # Change fid zone strings back to integer values
                    if zone_str.startswith('fid_'):
                        image_dict[zone_field] = int(zone_str[4:])
                    else:
                        image_dict[zone_field] = zone_str
                    image_dict['DATE'] = image_str
                    image_dict['LANDSAT'] = landsat.upper()
                    image_dict['PATH'] = path
                    image_dict['ROW'] = '000'
                    image_dict['SCENE_ID'] = '{}{}{}{}'.format(
                        image_dict['LANDSAT'], image_dict['PATH'],
                        image_dict['ROW'], image_dt.strftime('%Y%j'))
                    image_dict['YEAR'] = image_dt.year
                    image_dict['MONTH'] = image_dt.month
                    image_dict['DAY'] = image_dt.day
                    image_dict['DOY'] = int(image_dt.strftime('%j'))
                    # image_dict['PIXEL_COUNT'] = int(np.sum(zone_mask & mask_array))

                    # Save each row to a list
                    output_list.append(image_dict)

                # Append all rows for the year to a dataframe
                if not output_list:
                    logging.debug('    Empty output list, skipping')
                    continue
                output_df = output_df.append(output_list, ignore_index=True)
                output_df.sort_values(by=['DATE'], inplace=True)
                logging.debug('  {}'.format(output_path))
                output_df.to_csv(output_path, index=False, columns=landsat_daily_fields)


            # Combine/merge annual files into a single CSV
            logging.debug('\n  Merging annual Landsat CSV files')
            output_df = None
            for year in xrange(start_year, end_year + 1):
                # logging.debug('    {}'.format(year))
                input_path = os.path.join(
                    landsat_output_ws, '{}_landsat_{}.csv'.format(zone_str, year))
                try:
                    input_df = pd.read_csv(input_path)
                except:
                    continue
                try:
                    output_df = output_df.append(input_df)
                except:
                    output_df = input_df.copy()

            if output_df is not None and not output_df.empty:
                output_path = os.path.join(
                    zone_output_ws,
                    '{}_landsat_daily.csv'.format(zone_str))
                logging.debug('  {}'.format(output_path))
                output_df.sort_values(by=['DATE', 'ROW'], inplace=True)
                output_df.to_csv(
                    output_path, index=False, columns=landsat_daily_fields)


        if gridmet_flag:
            logging.info('  GRIDMET ETo/PPT')

            # Project the zone extent to the image OSR
            clip_extent = gdc.project_extent(
                zone_extent, zone_osr, gridmet_osr, zone_cs)
            logging.debug('  Extent: {}'.format(clip_extent))
            # clip_extent.buffer_extent(gridmet_cs)
            # logging.debug('  Extent: {}'.format(clip_extent))
            clip_extent.adjust_to_snap('EXPAND', gridmet_x, gridmet_y, gridmet_cs)
            logging.debug('  Extent: {}'.format(clip_extent))

            gridmet_images_ws = os.path.join(images_ws, gridmet_images_folder)
            if not os.path.isdir(gridmet_images_ws):
                logging.debug(
                    '  GRIDMET folder doesn\'t exist, skipping\n    {}'.format(
                        gridmet_images_ws))
                continue
            else:
                logging.info('  {}'.format(gridmet_images_ws))

            # Create an empty dataframe
            output_path = os.path.join(
                zone_output_ws,
                '{}_gridmet_monthly.csv'.format(zone_str))
            if os.path.isfile(output_path):
                if overwrite_flag:
                    logging.debug(
                        '  Output CSV already exists, removing\n    {}'.format(
                            output_path))
                    os.remove(output_path)
                else:
                    logging.debug(
                        '  Output CSV already exists, skipping\n    {}'.format(
                            output_path))
                    continue
            output_df = pd.DataFrame(columns=gridmet_monthly_fields)
            output_df[gridmet_int_fields] = output_df[gridmet_int_fields].astype(int)

            # Get list of all images
            image_list = [
                image for image in os.listdir(gridmet_images_ws)
                if gridmet_image_re.match(image)]
            dt_list = sorted(set([
                datetime.datetime(int(image[:4]), int(image[4:6]), 1)
                for image in image_list]))

            output_list = []
            for image_dt in dt_list:
                image_str = image_dt.date().isoformat()
                logging.debug('{}'.format(image_dt.date()))

                image_name_fmt = '{}_gridmet.{}.tif'.format(
                    image_dt.strftime('%Y%m'), '{}')

                # Save date specific properties
                image_dict = dict()

                # Workflow
                zs_list = [
                    ['eto', 'ETO'],
                    ['ppt', 'PPT'],
                ]
                for band_name, field in zs_list:
                    image_name = image_name_fmt.format(band_name)
                    logging.debug('  {} {}'.format(image_name, field))
                    if image_name not in image_list:
                        logging.debug('    Image doesn\'t exist, skipping')
                        continue
                    image_path = os.path.join(gridmet_images_ws, image_name)
                    # logging.debug('  {}'.format(image_path))

                    image_input_array, image_nodata = gdc.raster_to_array(
                        image_path, band=1, mask_extent=clip_extent,
                        fill_value=None, return_nodata=True)

                    # GRA_NearestNeighbour, GRA_Bilinear, GRA_Cubic,
                    #   GRA_CubicSpline
                    image_array = gdc.project_array(
                        image_input_array, gdal.GRA_NearestNeighbour,
                        gridmet_osr, gridmet_cs, clip_extent,
                        zone_osr, zone_cs, zone_extent,
                        output_nodata=None)
                    del image_input_array

                    # Skip fully masked zones
                    if (np.all(np.isnan(image_array)) or
                            np.all(image_array == image_nodata)):
                        logging.debug('    Empty array, skipping')
                        continue

                    image_dict[field] = np.mean(image_array[zone_mask])
                    del image_array

                if not image_dict:
                    logging.debug(
                        '    {} - no image data in zone, skipping'.format(
                            image_str))
                    continue

                # Save date specific properties
                # Change fid zone strings back to integer values
                if zone_str.startswith('fid_'):
                    image_dict[zone_field] = int(zone_str[4:])
                else:
                    image_dict[zone_field] = zone_str
                image_dict['DATE'] = image_str
                image_dict['YEAR'] = image_dt.year
                image_dict['MONTH'] = image_dt.month
                image_dict['WATER_YEAR'] = (image_dt + relativedelta(months=3)).year

                # Save each row to a list
                output_list.append(image_dict)

            # Append all rows for the year to a dataframe
            if not output_list:
                logging.debug('    Empty output list, skipping')
                continue
            output_df = output_df.append(output_list, ignore_index=True)
            output_df.sort_values(by=['DATE'], inplace=True)
            logging.debug('  {}'.format(output_path))
            output_df.to_csv(
                output_path, index=False, columns=gridmet_monthly_fields)


        if pdsi_flag:
            logging.info('  GRIDMET PDSI')
            logging.info('  Not currently implemented')


def get_ini_path(workspace):
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    ini_path = tkFileDialog.askopenfilename(
        initialdir=workspace, parent=root, filetypes=[('INI files', '.ini')],
        title='Select the target INI file')
    root.destroy()
    return ini_path


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Offline Zonal Statistics',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=lambda x: is_valid_file(parser, x),
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    else:
        args.ini = get_ini_path(os.getcwd())
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    zonal_stats(ini_path=args.ini, overwrite_flag=args.overwrite)
