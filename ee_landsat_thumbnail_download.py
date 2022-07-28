#--------------------------------
# Name:         ee_summary_thumbnails.py
# Purpose:      Generate summary tables
# Python:       3.6
#--------------------------------

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
from time import sleep
import urllib

# import io
# import Image
# from PIL import Image

import ee
import numpy as np
from osgeo import ogr
import pandas as pd

import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils


def main(ini_path=None, overwrite_flag=False):
    """Generate summary thumbnails

    Parameters
    ----------
    ini_path : str
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    """
    logging.info('\nGenerate summary thumbnails')

    # Inputs (eventually move to INI file?)
    vis_args = {
        'bands': ['red', 'green', 'blue'],
        # 'bands': ['swir1', 'nir', 'red'],
        'min': [0.01, 0.01, 0.01],
        'max': [0.4, 0.4, 0.4],
        'gamma': [1.8, 1.8, 1.8]
    }

    # Buffer zone polygon
    zone_buffer = 240

    # Generate images by DOY
    doy_flag = True
    # Generate images by date
    date_flag = True

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='SUMMARY')

    year_list = range(
        ini['INPUTS']['start_year'], ini['INPUTS']['end_year'] + 1)
    month_list = list(utils.wrapped_range(
        ini['INPUTS']['start_month'], ini['INPUTS']['end_month'], 1, 12))
    doy_list = list(utils.wrapped_range(
        ini['INPUTS']['start_doy'], ini['INPUTS']['end_doy'], 1, 366))

    # Add merged row XXX to keep list
    ini['INPUTS']['row_keep_list'].append('XXX')

    # Get ee features from shapefile
    zone_geom_list = gdc.shapefile_2_geom_list_func(
        ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
        reverse_flag=False)

    # Filter features by FID before merging geometries
    if ini['INPUTS']['fid_keep_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] not in ini['INPUTS']['fid_skip_list']]

    # Need zone_shp_path projection to build EE geometries
    zone = {}
    zone['osr'] = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone['proj'] = gdc.osr_wkt(zone['osr'])
    # zone['proj'] = ee.Projection(zone['proj']).wkt().getInfo()
    # zone['proj'] = zone['proj'].replace('\n', '').replace(' ', '')
    # logging.debug('  Zone Projection: {}'.format(zone['proj']))

    # Initialize Earth Engine API key
    logging.debug('')
    ee.Initialize()

    coll_dict = {
        'LT04': 'LANDSAT/LT04/C01/T1_SR',
        'LT05': 'LANDSAT/LT05/C01/T1_SR',
        'LE07': 'LANDSAT/LE07/C01/T1_SR',
        'LC08': 'LANDSAT/LC08/C01/T1_SR'
    }

    logging.info('\nProcessing zones')
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone['fid'] = zone_fid
        zone['name'] = zone_name.replace(' ', '_')
        zone['json'] = zone_json
        logging.info('ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))

        # Build EE geometry object for zonal stats
        zone['geom'] = ee.Geometry(
            geo_json=zone['json'], opt_proj=zone['proj'], opt_geodesic=False)
        # logging.debug('  Centroid: {}'.format(
        #     zone['geom'].centroid(100).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone['extent'] = gdc.Extent(
            ogr.CreateGeometryFromJson(json.dumps(zone['json'])).GetEnvelope())
        # zone['extent'] = gdc.Extent(zone['geom'].GetEnvelope())
        zone['extent'] = zone['extent'].ogrenv_swap()
        zone['extent'] = zone['extent'].buffer(zone_buffer)
        zone['extent'] = zone['extent'].adjust_to_snap(
            'EXPAND', ini['SPATIAL']['snap_x'], ini['SPATIAL']['snap_y'],
            ini['SPATIAL']['cellsize'])
        zone['geo'] = zone['extent'].geo(ini['SPATIAL']['cellsize'])
        zone['transform'] = gdc.geo_2_ee_transform(zone['geo'])
        # zone['transform'] = '[' + ','.join(map(str, zone['transform'])) + ']'
        zone['shape'] = zone['extent'].shape(ini['SPATIAL']['cellsize'])
        logging.debug('  Zone Shape: {}'.format(zone['shape']))
        logging.debug('  Zone Transform: {}'.format(zone['transform']))
        logging.debug('  Zone Extent: {}'.format(zone['extent']))
        # logging.debug('  Zone Geom: {}'.format(zone['geom'].getInfo()))

        # Build an EE geometry of the extent
        extent_geom = ee.Geometry.Rectangle(
            coords=list(zone['extent']), proj=zone['proj'], geodesic=False)

        if 'SUMMARY' in ini.keys():
            zone_output_ws = os.path.join(
                ini['SUMMARY']['output_ws'], zone['name'])
        elif 'EXPORT' in ini.keys():
            zone_output_ws = os.path.join(
                ini['EXPORT']['output_ws'], zone['name'])
        else:
            logging.error(
                'INI file does not contain a SUMMARY or EXPORT section')
            sys.exit()
        if not os.path.isdir(zone_output_ws):
            logging.debug('Folder {} does not exist, skipping'.format(
                zone_output_ws))
            continue

        landsat_daily_path = os.path.join(
            zone_output_ws, '{}_landsat_daily.csv'.format(zone['name']))
        if not os.path.isfile(landsat_daily_path):
            logging.error('  Landsat daily CSV does not exist, skipping zone')
            continue

        output_doy_ws = os.path.join(zone_output_ws, 'thumbnails_doy')
        output_date_ws = os.path.join(zone_output_ws, 'thumbnails_date')
        if overwrite_flag and os.path.isdir(output_doy_ws):
            for file_name in os.listdir(output_doy_ws):
                os.remove(os.path.join(output_doy_ws, file_name))
        if overwrite_flag and os.path.isdir(output_date_ws):
            for file_name in os.listdir(output_date_ws):
                os.remove(os.path.join(output_date_ws, file_name))
        if doy_flag and not os.path.isdir(output_doy_ws):
            os.makedirs(output_doy_ws)
        if date_flag and not os.path.isdir(output_date_ws):
            os.makedirs(output_date_ws)

        logging.debug('  Reading Landsat CSV')
        landsat_df = pd.read_csv(landsat_daily_path)
        # landsat_df = pd.read_csv(
        #     landsat_daily_path, parse_dates=['DATE'], index_col='DATE')
        landsat_df = landsat_df[landsat_df['PIXEL_COUNT'] > 0]

        # Common summary filtering
        logging.debug('  Filtering using INI SUMMARY parameters')
        if year_list:
            landsat_df = landsat_df[landsat_df['YEAR'].isin(year_list)]
        if month_list:
            landsat_df = landsat_df[landsat_df['MONTH'].isin(month_list)]
        if doy_list:
            landsat_df = landsat_df[landsat_df['DOY'].isin(doy_list)]

        if ini['INPUTS']['path_keep_list']:
            landsat_df = landsat_df[
                landsat_df['PATH'].isin(ini['INPUTS']['path_keep_list'])]
        if (ini['INPUTS']['row_keep_list'] and
                ini['INPUTS']['row_keep_list'] != ['XXX']):
            landsat_df = landsat_df[
                landsat_df['ROW'].isin(ini['INPUTS']['row_keep_list'])]

        # Assume the default is for these to be True and only filter if False
        if not ini['INPUTS']['landsat4_flag']:
            landsat_df = landsat_df[landsat_df['PLATFORM'] != 'LT04']
        if not ini['INPUTS']['landsat5_flag']:
            landsat_df = landsat_df[landsat_df['PLATFORM'] != 'LT05']
        if not ini['INPUTS']['landsat7_flag']:
            landsat_df = landsat_df[landsat_df['PLATFORM'] != 'LE07']
        if not ini['INPUTS']['landsat8_flag']:
            landsat_df = landsat_df[landsat_df['PLATFORM'] != 'LC08']

        if ini['INPUTS']['scene_id_keep_list']:
            # Replace XXX with primary ROW value for checking skip list SCENE_ID
            scene_id_df = pd.Series([
                s.replace('XXX', '{:03d}'.format(int(r)))
                for s, r in zip(landsat_df['SCENE_ID'], landsat_df['ROW'])])
            landsat_df = landsat_df[scene_id_df.isin(
                ini['INPUTS']['scene_id_keep_list']).values]
            # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
            # landsat_df = landsat_df[landsat_df['SCENE_ID'].isin(
            #     ini['INPUTS']['scene_id_keep_list'])]
        if ini['INPUTS']['scene_id_skip_list']:
            # Replace XXX with primary ROW value for checking skip list SCENE_ID
            scene_id_df = pd.Series([
                s.replace('XXX', '{:03d}'.format(int(r)))
                for s, r in zip(landsat_df['SCENE_ID'], landsat_df['ROW'])])
            landsat_df = landsat_df[np.logical_not(scene_id_df.isin(
                ini['INPUTS']['scene_id_skip_list']).values)]
            # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
            # landsat_df = landsat_df[np.logical_not(landsat_df['SCENE_ID'].isin(
            #     ini['INPUTS']['scene_id_skip_list']))]

        # Filter by QA/QC value
        if ini['SUMMARY']['max_qa'] >= 0 and not landsat_df.empty:
            logging.debug('    Maximum QA: {0}'.format(
                ini['SUMMARY']['max_qa']))
            landsat_df = landsat_df[landsat_df['QA'] <= ini['SUMMARY']['max_qa']]

        # Filter by average cloud score
        if ini['SUMMARY']['max_cloud_score'] < 100 and not landsat_df.empty:
            logging.debug('    Maximum cloud score: {0}'.format(
                ini['SUMMARY']['max_cloud_score']))
            landsat_df = landsat_df[
                landsat_df['CLOUD_SCORE'] <= ini['SUMMARY']['max_cloud_score']]

        # Filter by Fmask percentage
        if ini['SUMMARY']['max_fmask_pct'] < 100 and not landsat_df.empty:
            landsat_df['FMASK_PCT'] = 100 * (
                landsat_df['FMASK_COUNT'] / landsat_df['FMASK_TOTAL'])
            logging.debug('    Max Fmask threshold: {}'.format(
                ini['SUMMARY']['max_fmask_pct']))
            landsat_df = landsat_df[
                landsat_df['FMASK_PCT'] <= ini['SUMMARY']['max_fmask_pct']]

        # Filter low count SLC-off images
        if ini['SUMMARY']['min_slc_off_pct'] > 0 and not landsat_df.empty:
            logging.debug('    Mininum SLC-off threshold: {}%'.format(
                ini['SUMMARY']['min_slc_off_pct']))
            # logging.debug('    Maximum pixel count: {}'.format(
            #     max_pixel_count))
            slc_off_mask = (
                (landsat_df['PLATFORM'] == 'LE07') &
                ((landsat_df['YEAR'] >= 2004) |
                 ((landsat_df['YEAR'] == 2003) & (landsat_df['DOY'] > 151))))
            slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / landsat_df['PIXEL_TOTAL'])
            # slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / max_pixel_count)
            landsat_df = landsat_df[
                ((slc_off_pct >= ini['SUMMARY']['min_slc_off_pct']) & slc_off_mask) |
                (~slc_off_mask)]

        if landsat_df.empty:
            logging.error(
                '  Empty Landsat dataframe after filtering, skipping zone')
            continue


        logging.debug('  Downloading thumbnails')
        for landsat, start_date in zip(landsat_df['PLATFORM'], landsat_df['DATE']):
            start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = start_dt + datetime.timedelta(days=1)
            end_date = end_dt.strftime('%Y-%m-%d')

            output_doy_path = os.path.join(
                output_doy_ws,
                '{}_{}.png'.format(start_dt.strftime('%j_%Y-%m-%d'), landsat))
            output_date_path = os.path.join(
                output_date_ws,
                '{}_{}.png'.format(start_dt.strftime('%Y-%m-%d_%j'), landsat))

            # DEADBEEF - This seems like a poor approach
            save_doy_flag = False
            save_date_flag = False
            if doy_flag and not os.path.isfile(output_doy_path):
                save_doy_flag = True
            if date_flag and not os.path.isfile(output_date_path):
                save_date_flag = True
            if not save_doy_flag and not save_date_flag:
                logging.debug('  {} - file already exists, skipping'.format(
                    start_date))
                continue
            logging.debug('  {}'.format(start_date))
            # logging.debug('  {}'.format(output_path))

            if landsat in ['LT04', 'LT05', 'LE07']:
                ee_coll = ee.ImageCollection(coll_dict[landsat]).select(
                    ['B1', 'B2', 'B3', 'B4', 'B5', 'B7'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
            elif landsat in ['LC08']:
                ee_coll = ee.ImageCollection(coll_dict[landsat]).select(
                    ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
            ee_coll = ee_coll.filterDate(start_date, end_date)
            ee_image = ee.Image(ee_coll.median().divide(10000)) \
                .visualize(**vis_args) \
                .reproject(crs=zone['proj'], crsTransform=zone['transform']) \
                .paint(zone['geom'], color=0.5, width=1) \
                .clip(extent_geom)

            # Get the image thumbnail
            for i in range(10):
                try:
                    output_url = ee_image.getThumbUrl({'format': 'png'})
                    break
                except Exception as e:
                    logging.error('  Exception: {}, retry {}'.format(e, i))
                    logging.debug('{}'.format(e))
                    sleep(i ** 2)

            for i in range(10):
                try:
                    # DEADBEEF - This seems like a poor approach
                    if save_doy_flag and save_date_flag:
                        urllib.request.urlretrieve(output_url, output_doy_path)
                        shutil.copy(output_doy_path, output_date_path)
                    elif save_doy_flag:
                        urllib.request.urlretrieve(output_url, output_doy_path)
                    elif save_date_flag:
                        urllib.request.urlretrieve(output_url, output_date_path)
                    sleep(0.1)
                    break
                except Exception as e:
                    logging.error('  Exception: {}, retry {}'.format(e, i))
                    logging.debug('{}'.format(e))
                    sleep(i ** 2)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate summary thumbnails',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    else:
        args.ini = utils.get_ini_path(os.getcwd())
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.getLogger('googleapiclient').setLevel(logging.ERROR)
    logging.info('\n{0}'.format('#' * 80))
    log_f = '{0:<20s} {1}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, overwrite_flag=args.overwrite)
