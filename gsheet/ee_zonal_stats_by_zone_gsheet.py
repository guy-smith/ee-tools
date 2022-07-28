#--------------------------------
# Name:         ee_zonal_stats_by_zone_gsheet.py
# Purpose:      Download zonal stats by zone using Earth Engine
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
from collections import defaultdict
import datetime
from io import StringIO
import json
import logging
import math
import os
import pprint
import re
import requests
from subprocess import check_output
import sys

import ee
import gspread
from oauth2client import service_account
# import numpy as np
from osgeo import ogr
import pandas as pd

# This is an awful way of getting the parent folder into the path
# We really should package this up as a module with a setup.py
# This way the ee_tools folders would be in the
#   PYTHONPATH env. variable
ee_tools_path = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(ee_tools_path, 'ee_tools'))
sys.path.insert(0, ee_tools_path)
import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils

pp = pprint.PrettyPrinter(indent=4)

SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'MapWater-4e2df36b1209.json'


def main(ini_path=None, overwrite_flag=False):
    """Earth Engine Zonal Stats Export

    Parameters
    ----------
    ini_path : str
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    """
    logging.info('\nEarth Engine zonal statistics by zone')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='ZONAL_STATS')

    if ini['EXPORT']['export_dest'].lower() != 'gsheet':
        logging.critical(
            '\nERROR: Only GSheet exports are currently supported\n')
        sys.exit()

    # These may eventually be set in the INI file
    landsat_daily_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'SCENE_ID', 'PLATFORM',
        'PATH', 'ROW', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'AREA', 'PIXEL_SIZE', 'PIXEL_COUNT', 'PIXEL_TOTAL',
        'FMASK_COUNT', 'FMASK_TOTAL', 'FMASK_PCT', 'CLOUD_SCORE', 'QA']

    # Concert REFL_TOA, REFL_SUR, and TASSELED_CAP products to bands
    if 'refl_toa' in ini['ZONAL_STATS']['landsat_products']:
        ini['ZONAL_STATS']['landsat_products'].extend([
            'blue_toa', 'green_toa', 'red_toa',
            'nir_toa', 'swir1_toa', 'swir2_toa'])
        ini['ZONAL_STATS']['landsat_products'].remove('refl_toa')
    if 'refl_sur' in ini['ZONAL_STATS']['landsat_products']:
        ini['ZONAL_STATS']['landsat_products'].extend([
            'blue_sur', 'green_sur', 'red_sur',
            'nir_sur', 'swir1_sur', 'swir2_sur'])
        ini['ZONAL_STATS']['landsat_products'].remove('refl_sur')
    if 'tasseled_cap' in ini['ZONAL_STATS']['landsat_products']:
        ini['ZONAL_STATS']['landsat_products'].extend([
            'tc_bright', 'tc_green', 'tc_wet'])
        ini['ZONAL_STATS']['landsat_products'].remove('tasseled_cap')
    landsat_daily_fields.extend(
        [p.upper() for p in ini['ZONAL_STATS']['landsat_products']])

    # DEADBEEF - Hack to get Beamer ETStar threshold
    if 'etstar_mean' in ini['ZONAL_STATS']['landsat_products']:
        inputs.parse_section(ini, section='BEAMER')
        landsat_daily_fields.insert(
            landsat_daily_fields.index('CLOUD_SCORE'), 'ETSTAR_COUNT')

    # # Concert REFL_TOA, REFL_SUR, and TASSELED_CAP products to bands
    # # if 'tmean' in ini['ZONAL_STATS']['gridmet_products']:
    # #     ini['ZONAL_STATS']['gridmet_products'].extend(['tmin', 'tmax'])
    # gridmet_daily_fields.extend(
    #     [p.upper() for p in ini['ZONAL_STATS']['gridmet_products']])
    # gridmet_monthly_fields.extend(
    #     [p.upper() for p in ini['ZONAL_STATS']['gridmet_products']])

    # Convert the shapefile to geojson
    # if not os.path.isfile(ini['ZONAL_STATS']['zone_geojson']):
    if not os.path.isfile(ini['ZONAL_STATS']['zone_geojson']) or overwrite_flag:
        logging.info('\nConverting zone shapefile to GeoJSON')
        logging.debug('  {}'.format(ini['ZONAL_STATS']['zone_geojson']))
        check_output([
            'ogr2ogr', '-f', 'GeoJSON', '-preserve_fid',
            '-select', '{}'.format(ini['INPUTS']['zone_field']),
            # '-lco', 'COORDINATE_PRECISION=2'
            ini['ZONAL_STATS']['zone_geojson'],
            ini['INPUTS']['zone_shp_path']])

    # # Get ee features from shapefile
    # zone_geom_list = gdc.shapefile_2_geom_list_func(
    #     ini['INPUTS']['zone_shp_path'],
    #     zone_field=ini['INPUTS']['zone_field'],
    #     reverse_flag=False)
    # # zone_count = len(zone_geom_list)
    # # output_fmt = '_{0:0%sd}.csv' % str(int(math.log10(zone_count)) + 1)

    # Read in the zone geojson
    logging.debug('\nReading zone GeoJSON')
    try:
        with open(ini['ZONAL_STATS']['zone_geojson'], 'r') as f:
            zones = json.load(f)
    except Exception as e:
        logging.error('  Error reading zone geojson file, removing')
        logging.debug('  Exception: {}'.format(e))
        os.remove(ini['ZONAL_STATS']['zone_geojson'])

    # Check if the zone_names are unique
    # Eventually support merging common zone_names
    zone_names = [
        str(z['properties'][ini['INPUTS']['zone_field']]) \
            .replace(' ', '_').lower()
        for z in zones['features']]
    if len(set(zone_names)) != len(zones['features']):
        logging.error(
            '\nERROR: There appear to be duplicate zone ID/name values.'
            '\n  Currently, the values in "{}" must be unique.'
            '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
        return False

    # # Check if the zone_names are unique
    # # Eventually support merging common zone_names
    # if len(set([z[1] for z in zone_geom_list])) != len(zone_geom_list):
    #     logging.error(
    #         '\nERROR: There appear to be duplicate zone ID/name values.'
    #         '\n  Currently, the values in "{}" must be unique.'
    #         '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
    #     return False

    # Get projection from shapefile to build EE geometries
    # GeoJSON technically should always be EPSG:4326 so don't assume
    #  coordinates system property will be set
    zone_osr = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone_wkt = gdc.osr_wkt(zone_osr)

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zone_osr, ini['SPATIAL']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zone_osr))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.warning('  Zone Proj4:   {}'.format(
            zone_osr.ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['SPATIAL']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(zone_wkt))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['SPATIAL']['cellsize']))

    # Authenticate with Google Sheets API
    logging.debug('\nAuthenticating with Google Sheets API')
    oauth_cred = service_account.ServiceAccountCredentials.from_json_keyfile_name(
        CLIENT_SECRET_FILE, SCOPES)
    gsheet_cred = gspread.authorize(oauth_cred)

    # Initialize Earth Engine API key
    logging.info('\nInitializing Earth Engine')
    ee.Initialize()
    utils.ee_request(ee.Number(1).getInfo())

    # Get current running tasks before getting file lists
    tasks = utils.get_ee_tasks()

    # Build separate tile lists for each zone
    # Build tile lists before filtering by FID below
    # DEADBEEF - This is a list of all "possible" tile that is
    #   independent of the INI tile settings
    ini['ZONAL_STATS']['zone_tile_json'] = {}
    ini['ZONAL_STATS']['tile_scene_json'] = {}
    # if os.path.isfile(ini['ZONAL_STATS']['zone_tile_path']):
    if (os.path.isfile(ini['ZONAL_STATS']['zone_tile_path']) and
            not overwrite_flag):
        logging.debug('\nReading zone tile lists\n  {}'.format(
            ini['ZONAL_STATS']['zone_tile_path']))
        with open(ini['ZONAL_STATS']['zone_tile_path'], 'r') as f:
            ini['ZONAL_STATS']['zone_tile_json'] = json.load(f)
    else:
        logging.info('\nBuilding zone tile lists')
        step = 1000
        zone_n = len(zones['features'])
        for zone_i in range(0, len(zones['features']), step):
            logging.debug('  Zones: {}-{}'.format(
                zone_i, min(zone_i + step, zone_n) - 1))
            zone_ftr_sub = zones['features'][zone_i: min(zone_i + step, zone_n)]

            # Build the zones feature collection in a list comprehension
            #   in order to set the correct spatial reference
            zone_field = ini['INPUTS']['zone_field']
            zone_coll = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry(f['geometry'], zone_wkt, False),
                    {zone_field: f['properties'][zone_field]})
                for f in zone_ftr_sub])

            # Load the WRS2 custom footprint collection
            tile_field = 'WRS2_TILE'
            wrs2_coll = ee.FeatureCollection(
                    'projects/usgs-ssebop/wrs2_descending_custom') \
                .filterBounds(zone_coll.geometry())

            # Extract tile values from joined collection
            def ftr_property(ftr):
                scenes = ee.FeatureCollection(ee.List(ee.Feature(ftr).get('scenes'))) \
                    .toList(100).map(lambda tile: ee.Feature(tile).get(tile_field))
                return ee.Feature(None, {
                    zone_field: ee.String(ftr.get(zone_field)),
                    tile_field: scenes})

            # Intersect the geometry and wrs2 collections
            spatialFilter = ee.Filter.intersects(
                leftField='.geo', rightField='.geo', maxError=10)
            join_coll = ee.FeatureCollection(
                ee.Join.saveAll(matchesKey='scenes') \
                    .apply(zone_coll, wrs2_coll, spatialFilter) \
                    .map(ftr_property))

            # Build a list of tiles for each zone
            for f in utils.ee_getinfo(join_coll)['features']:
                zone_name = str(f['properties'][ini['INPUTS']['zone_field']]) \
                    .replace(' ', '_')
                ini['ZONAL_STATS']['zone_tile_json'][zone_name] = sorted(list(set(
                    f['properties'][tile_field])))

        logging.debug('  Saving zone tile dictionary')
        logging.debug('    {}'.format(ini['ZONAL_STATS']['zone_tile_path']))
        with open(ini['ZONAL_STATS']['zone_tile_path'], 'w') as f:
            json.dump(ini['ZONAL_STATS']['zone_tile_json'], f, sort_keys=True)

    # Filter features by FID
    # Don't filter until after tile lists are built
    if ini['INPUTS']['fid_keep_list']:
        zones['features'] = [
            ftr for ftr in zones['features']
            if ftr['id'] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zones['features'] = [
            ftr for ftr in zones['features']
            if ftr['id'] not in ini['INPUTS']['fid_skip_list']]

    # Merge geometries (after filtering by FID above)
    if ini['INPUTS']['merge_geom_flag']:
        logging.debug('\nMerging geometries')
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone_ftr in zones['features']:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone_ftr['geometry'])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        zones['features'] = [{
            'type': 'Feature',
            'id': 0,
            'properties': {ini['INPUTS']['zone_field']: zones['name']},
            'geometry': json.loads(merge_geom.ExportToJson())}]

        # Collapse WRS2 tile lists for merged geometry
        ini['ZONAL_STATS']['zone_tile_json'][zones['name']] = sorted(list(set([
            pr for pr_list in ini['ZONAL_STATS']['zone_tile_json'].values()
            for pr in pr_list])))
        logging.debug('  WRS2 Tiles: {}'.format(
            ini['ZONAL_STATS']['zone_tile_json'][zones['name']]))

    # Get end date of GRIDMET (if needed)
    # This could be moved to inside the INI function
    if (ini['ZONAL_STATS']['gridmet_monthly_flag'] or
            ini['ZONAL_STATS']['gridmet_daily_flag']):
        gridmet_end_dt = utils.ee_request(ee.Date(ee.Image(
            ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(
                    '{}-01-01'.format(ini['INPUTS']['end_year'] - 1),
                    '{}-01-01'.format(ini['INPUTS']['end_year'] + 1)) \
                .limit(1, 'system:time_start', False) \
                .first()
            ).get('system:time_start')).format('YYYY-MM-dd')).getInfo()
        gridmet_end_dt = datetime.datetime.strptime(
            gridmet_end_dt, '%Y-%m-%d')
        logging.debug('    Last GRIDMET date: {}'.format(gridmet_end_dt))


    # Calculate zonal stats for each feature separately
    logging.info('')
    # for zone_fid, zone_name, zone_json in zone_geom_list:
    #     zone['fid'] = zone_fid
    #     zone['name'] = zone_name.replace(' ', '_')
    #     zone['json'] = zone_json
    for zone_ftr in zones['features']:
        zone = {}
        zone['fid'] = zone_ftr['id']
        zone['name'] = str(zone_ftr['properties'][ini['INPUTS']['zone_field']]) \
            .replace(' ', '_')
        zone['json'] = zone_ftr['geometry']

        logging.info('ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))
        logging.debug('  Zone')

        # Build EE geometry object for zonal stats
        zone['geom'] = ee.Geometry(
            geo_json=zone['json'], opt_proj=zone_wkt, opt_geodesic=False)
        # logging.debug('  Centroid: {}'.format(
        #     zone['geom'].centroid(1).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone_geom = ogr.CreateGeometryFromJson(json.dumps(zone['json']))
        if zone_geom.GetGeometryName() in ['POINT', 'MULTIPOINT']:
            # Compute area as cellsize * number of points
            point_count = 0
            for i in range(0, zone_geom.GetGeometryCount()):
                point_count += zone_geom.GetGeometryRef(i).GetPointCount()
            zone['area'] = ini['SPATIAL']['cellsize'] * point_count
        else:
            # Adjusting area up to nearest multiple of cellsize to account for
            #   polygons that were modified to avoid interior holes
            zone['area'] = ini['SPATIAL']['cellsize'] * math.ceil(
                zone_geom.GetArea() / ini['SPATIAL']['cellsize'])

        # zone['area'] = zone_geom.GetArea()
        zone['extent'] = gdc.Extent(zone_geom.GetEnvelope())
        # zone['extent'] = gdc.Extent(zone['geom'].GetEnvelope())
        zone['extent'] = zone['extent'].ogrenv_swap()
        zone['extent'] = zone['extent'].adjust_to_snap(
            'EXPAND', ini['SPATIAL']['snap_x'], ini['SPATIAL']['snap_y'],
            ini['SPATIAL']['cellsize'])
        zone['geo'] = zone['extent'].geo(ini['SPATIAL']['cellsize'])
        zone['transform'] = gdc.geo_2_ee_transform(zone['geo'])
        # zone['transform'] = '[' + ','.join(map(str, zone['transform'])) + ']'
        zone['shape'] = zone['extent'].shape(ini['SPATIAL']['cellsize'])
        logging.debug('    Zone Shape: {}'.format(zone['shape']))
        logging.debug('    Zone Transform: {}'.format(zone['transform']))
        logging.debug('    Zone Extent: {}'.format(zone['extent']))
        # logging.debug('    Zone Geom: {}'.format(zone['geom'].getInfo()))

        # Assume all pixels in all images could be reduced
        zone['max_pixels'] = zone['shape'][0] * zone['shape'][1]
        logging.debug('    Max Pixels: {}'.format(zone['max_pixels']))

        # Set output spatial reference
        # Eventually allow user to manually set these
        # output_crs = zone['proj']
        ini['EXPORT']['transform'] = zone['transform']
        logging.debug('    Output Projection: {}'.format(
            ini['SPATIAL']['crs']))
        logging.debug('    Output Transform: {}'.format(
            ini['EXPORT']['transform']))

        zone['output_ws'] = os.path.join(
            ini['ZONAL_STATS']['output_ws'], zone['name'])
        if not os.path.isdir(zone['output_ws']):
            os.makedirs(zone['output_ws'])

        if ini['ZONAL_STATS']['landsat_flag']:
            landsat_func(
                gsheet_cred, landsat_daily_fields, ini, zone, tasks,
                overwrite_flag)


def landsat_func(gsheet_cred, export_fields, ini, zone, tasks,
                 overwrite_flag=False):
    """

    Function will attempt to generate export tasks only for missing SCENE_IDs
    Also try to limit the products to only those with missing data

    Parameters
    ----------
    gsheet_cred :
    export_fields :
    ini : dict
        Input file parameters.
    zone : dict
        Zone specific parameters.
    tasks :
    overwrite_flag : bool, optional
        If True, overwrite existing values (the default is False).
        Don't remove/replace the CSV file directly.

    """
    logging.info('  Landsat')

    landsat_products = ini['ZONAL_STATS']['landsat_products'][:]
    landsat_fields = [f.upper() for f in landsat_products]

    # Assuming Google Sheet exists and has the target columns
    # Assuming worksheet is called "Landsat_Daily"
    logging.info('    Reading Landsat GSHEET')
    gsheet = gsheet_cred.open_by_key(ini['GSHEET']['gsheet_id'])\
        .worksheet(ini['GSHEET']['landsat_daily'])
    output_fields = gsheet.row_values(1)

    # # DEADBEEF - Changing the field structure should probably be done manually
    # # Add missing fields
    # for field_name in export_fields:
    #     if field_name not in output_fields:
    #         logging.info('    Adding field: {}'.format(field_name))
    #         gsheet.add_cols(1)
    #         gsheet.update_cell(1, gsheet.col_count, field_name)
    #         output_fields.append(field_name)

    # Read in full data frame
    input_df = pd.DataFrame(
        data=gsheet.get_all_values()[1:], columns=output_fields)
    input_df.set_index(['ZONE_NAME', 'SCENE_ID'], inplace=True, drop=True)
    print(input_df)
    input('ENTER')

    def gsheet_writer(output_df):
        """Write (append) dataframe to Google Sheet"""
        temp_df = output_df.copy()
        temp_df.reset_index(drop=False, inplace=True)

        output_rows = len(output_df)
        gsheet.add_rows(output_rows)
        cell_range = '{ul}:{lr}'.format(
            ul=gspread.utils.rowcol_to_a1(gsheet.row_count, 1),
            lr=gspread.utils.rowcol_to_a1(
                gsheet.row_count + output_rows, len(output_fields)))
        cell_list = gsheet.range(cell_range)
        for cell in cell_list:
            print(cell.col, cell.row, output_fields[cell.col-1])
            input('ENTER')
            # Assuming the cells are in the same order as the fields
            # print(gsheet_fields[cell.col-1])
            # print(temp_df[gsheet_fields[cell.col-1]])
            # cell.value = temp_df[gsheet_fields[cell.col-1]]
        # output_gs.update_cells(cell_list)

    # Pre-filter by tile
    # First get the list of possible tiles for each zone
    try:
        zone_tile_list = ini['ZONAL_STATS']['zone_tile_json'][zone['name']]
    except KeyError:
        logging.info('    No matching tiles, skipping zone')
        return True
    if ini['INPUTS']['path_keep_list']:
        zone_tile_list = [
            tile for tile in zone_tile_list
            if int(tile[1:4]) in ini['INPUTS']['path_keep_list']]
    if zone_tile_list and ini['INPUTS']['row_keep_list']:
        zone_tile_list = [
            tile for tile in zone_tile_list
            if int(tile[5:8]) in ini['INPUTS']['row_keep_list']]
    if zone_tile_list and ini['INPUTS']['tile_keep_list']:
        zone_tile_list = [
            tile for tile in zone_tile_list
            if tile in ini['INPUTS']['tile_keep_list']]
    if not zone_tile_list:
        logging.info('    No matching tiles, skipping zone')
        return True

    # Initialize the Landsat object
    # For getting SCENE_ID lists, don't use zone_geom or products
    #   and set mosaic_method to 'none' to get separate SCENE_ID lists
    #   for each tile
    # These will be applied below
    landsat_args = {
        k: v for section in ['INPUTS']
        for k, v in ini[section].items()
        if k in [
            'landsat4_flag', 'landsat5_flag',
            'landsat7_flag', 'landsat8_flag',
            'fmask_flag', 'acca_flag',
            'start_year', 'end_year',
            'start_month', 'end_month',
            'start_doy', 'end_doy',
            'scene_id_keep_list', 'scene_id_skip_list',
            'path_keep_list', 'row_keep_list',
            'refl_sur_method', 'adjust_method', 'mosaic_method']}
    landsat = ee_common.Landsat(landsat_args)
    if ini['INPUTS']['tile_geom']:
        landsat.tile_geom = ini['INPUTS']['tile_geom']

    # DEADBEEF - Drop paths that aren't in the full zone_tile_list
    #   Intentionally using non-filtered zone tile list
    zone_path_list = sorted(list(set([
        int(tile[1:4])
        for tile in ini['ZONAL_STATS']['zone_tile_json'][zone['name']]])))
    # if not output_df.empty and (~output_df['PATH'].isin(zone_path_list)).any():
    #     logging.info('    Removing invalid path entries')
    #     output_df = output_df[output_df['PATH'].isin(zone_path_list)]
    #     gsheet_writer(output_df)

    # # # DEADBEEF - Remove old Landsat 8 data
    # # l8_pre_mask = (
    # #     (output_df['PLATFORM'] == 'LC08') & (output_df['YEAR'] >= 2013) &
    # #     (output_df['YEAR'] <= 2014))
    # # if not output_df.empty and l8_pre_mask.any():
    # #     logging.info('    Removing old Landsat 8 entries')
    # #     # logging.info('    {}'.format(output_df[drop_mask]['SCENE_ID'].values))
    # #     output_df.drop(output_df[l8_pre_mask].index, inplace=True)
    # #     csv_writer(output_df, output_path, output_fields)
    # #     # input('ENTER')

    # # # DEADBEEF - Look for duplicate SCENE_IDs
    # # if not output_df.empty and output_df.duplicated(['SCENE_ID']).any():
    # #     logging.debug('    Removing duplicate SCENE_IDs')
    # #     output_df = output_df[output_df.duplicated(['SCENE_ID'], False)]

    # # # DEADBEEF - Remove entries with nodata (but PIXEL_COUNT > 0)
    # # drop_mask = (
    # #     output_df['NDVI_TOA'].isnull() & (output_df['PIXEL_COUNT'] > 0))
    # # if not output_df.empty and drop_mask.any():
    # #     logging.info('    Removing bad PIXEL_COUNT entries')
    # #     # logging.info('    {}'.format(output_df[drop_mask]['SCENE_ID'].values))
    # #     output_df.drop(output_df[drop_mask].index, inplace=True)
    # #     csv_writer(output_df, output_path, output_fields)

    # # # DEADBEEF - Remove all old style empty entries
    # # drop_mask = (
    # #     output_df['NDVI_TOA'].isnull() & (output_df['PIXEL_TOTAL'] > 0))
    # # if not output_df.empty and drop_mask.any():
    # #     logging.debug('    Removing old empty entries')
    # #     output_df.drop(output_df[drop_mask].index, inplace=True)
    # #     csv_writer(output_df, output_path, output_fields)

    # # # DEADBEEF - Remove all empty entries (for testing filling code)
    # # drop_mask = output_df['NDVI_TOA'].isnull()
    # # if not output_df.empty and drop_mask.any():
    # #     logging.debug('    Removing all empty entries')
    # #     output_df.drop(output_df[drop_mask].index, inplace=True)
    # #     csv_writer(output_df, output_path, output_fields)
    # #     input('ENTER')

    # Use the SCENE_ID as the index
    output_df.set_index('SCENE_ID', inplace=True, drop=True)
    # output_df.index.name = 'SCENE_ID'

    # Filter based on the pre-computed SCENE_ID lists from the init
    # Get the list of possible SCENE_IDs for each zone tile
    logging.debug('    Getting SCENE_ID lists')
    export_ids = set()
    for tile in zone_tile_list:
        if tile in ini['ZONAL_STATS']['tile_scene_json'].keys():
            # Read the previously computed tile SCENE_ID list
            export_ids.update(
                ini['ZONAL_STATS']['tile_scene_json'][tile])
        else:
            # Compute the SCENE_ID list for each tile if needed
            logging.debug('      {}'.format(tile))
            path_row_re = re.compile('p(?P<PATH>\d{1,3})r(?P<ROW>\d{1,3})')
            path, row = list(map(int, path_row_re.match(tile).groups()))

            # Filter the Landsat collection down to a single tile
            landsat.zone_geom = None
            landsat.products = []
            landsat.mosaic_method = 'none'
            landsat.path_keep_list = [path]
            landsat.row_keep_list = [row]
            landsat_coll = landsat.get_collection()

            # Get new scene ID list
            ini['ZONAL_STATS']['tile_scene_json'][tile] = utils.ee_getinfo(
                landsat_coll.aggregate_histogram('SCENE_ID'))
            export_ids.update(ini['ZONAL_STATS']['tile_scene_json'][tile])

    # If export_ids is empty, all SCENE_IDs may have been filtered
    if not export_ids:
        logging.info(
            '    No SCENE_IDs to process after applying INI filters, '
            'skipping zone')
        return False

    # Compute mosaiced SCENE_IDs after filtering
    if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
        mosaic_id_dict = defaultdict(list)
        for scene_id in export_ids:
            mosaic_id = '{}XXX{}'.format(scene_id[:8], scene_id[11:])
            mosaic_id_dict[mosaic_id].append(scene_id)
        export_ids = set(mosaic_id_dict.keys())

    # For overwrite, drop all expected entries from existing output DF
    if overwrite_flag:
        output_df = output_df[~output_df.index.isin(list(export_ids))]

    # # # # DEADBEEF - Reset zone area
    # # if not output_df.empty:
    # #     logging.info('    Updating zone area')
    # #     output_df.loc[
    # #         output_df.index.isin(list(export_ids)),
    # #         ['AREA']] = zone['area']
    # #     csv_writer(output_df, output_path, output_fields)

    # List of SCENE_IDs that are entirely missing
    # This may include scenes that don't intersect the zone
    missing_all_ids = export_ids - set(output_df.index.values)
    # logging.info('  Dates missing all values: {}'.format(
    #     ', '.join(sorted(missing_all_ids))))

    # If there are any fully missing scenes, identify whether they
    #   intersect the zone or not
    # Exclude non-intersecting SCENE_IDs from export_ids set
    # Add non-intersecting SCENE_IDs directly to the output dataframe
    if missing_all_ids:
        # Get SCENE_ID list mimicking a full extract below
        #   (but without products)
        # Start with INI path/row keep list but update based on SCENE_ID later
        landsat.products = []
        landsat.path_keep_list = landsat_args['path_keep_list']
        landsat.row_keep_list = landsat_args['row_keep_list']
        landsat.zone_geom = zone['geom']
        landsat.mosaic_method = landsat_args['mosaic_method']

        # Only use the scene_id_keep_list if there was already data in the DF
        if output_df.index.values.any():
            # SCENE_ID keep list must be non-mosaiced IDs
            if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
                landsat.scene_id_keep_list = sorted(set([
                    scene_id for mosaic_id in missing_all_ids
                    for scene_id in mosaic_id_dict[mosaic_id]]))
            else:
                landsat.scene_id_keep_list = sorted(missing_all_ids)
            landsat.set_landsat_from_scene_id()
            landsat.set_tiles_from_scene_id()

        # Get the SCENE_IDs that intersect the zone
        # Process each Landsat type independently
        # Was having issues getting the full scene list at once
        logging.debug('    Getting intersecting SCENE_IDs')
        missing_zone_ids = set()
        landsat_type_list = landsat._landsat_list[:]
        for landsat_str in landsat_type_list:
            landsat._landsat_list = [landsat_str]
            missing_zone_ids.update(set(utils.ee_getinfo(
                landsat.get_collection().aggregate_histogram('SCENE_ID'))))
            logging.debug('      {} {}'.format(
                landsat_str, len(missing_zone_ids)))
        landsat._landsat_list = landsat_type_list

        # # Get the SCENE_IDs that intersect the zone
        # logging.debug('    Getting intersecting SCENE_IDs')
        # missing_zone_ids = set(utils.ee_getinfo(
        #     landsat.get_collection().aggregate_histogram('SCENE_ID')))

        # Difference of sets are SCENE_IDs that don't intersect
        missing_skip_ids = missing_all_ids - missing_zone_ids

        # Updating missing all SCENE_ID list to not include
        #   non-intersecting scenes
        missing_all_ids = set(missing_zone_ids)

        # Remove skipped/empty SCENE_IDs from possible SCENE_ID list
        export_ids = export_ids - missing_skip_ids
        # logging.debug('  Missing Include: {}'.format(
        #     ', '.join(sorted(missing_zone_ids))))
        # logging.debug('  Missing Exclude: {}'.format(
        #     ', '.join(sorted(missing_skip_ids))))
        logging.info('    Include ID count: {}'.format(
            len(missing_zone_ids)))
        logging.info('    Exclude ID count: {}'.format(
            len(missing_skip_ids)))

        if missing_skip_ids:
            logging.debug('    Appending empty non-intersecting SCENE_IDs')
            missing_df = pd.DataFrame(
                index=missing_skip_ids, columns=output_df.columns)
            missing_df.index.name = 'SCENE_ID'
            missing_df['ZONE_NAME'] = str(zone['name'])
            missing_df['ZONE_FID'] = zone['fid']
            missing_df['AREA'] = zone['area']
            missing_df['PLATFORM'] = missing_df.index.str.slice(0, 4)
            missing_df['PATH'] = missing_df.index.str.slice(5, 8).astype(int)
            missing_df['DATE'] = pd.to_datetime(
                missing_df.index.str.slice(12, 20), format='%Y%m%d')
            missing_df['YEAR'] = missing_df['DATE'].dt.year
            missing_df['MONTH'] = missing_df['DATE'].dt.month
            missing_df['DAY'] = missing_df['DATE'].dt.day
            missing_df['DOY'] = missing_df['DATE'].dt.dayofyear.astype(int)
            # DEADBEEF - Non-intersecting ROW values
            #   Does it matter what value is used here?
            #   We don't know the dominate or any ROW value here
            #   It can't be XXX since the column type is int
            #   Setting to np.nan causes issues in summary_tables (and qaqc)
            missing_df['ROW'] = 0
            missing_df['QA'] = np.nan
            # missing_df['QA'] = 0
            missing_df['PIXEL_SIZE'] = landsat.cellsize
            missing_df['PIXEL_COUNT'] = 0
            missing_df['PIXEL_TOTAL'] = 0
            missing_df['FMASK_COUNT'] = 0
            missing_df['FMASK_TOTAL'] = 0
            missing_df['FMASK_PCT'] = np.nan
            if 'etstar_mean' in landsat.products:
                missing_df['ETSTAR_COUNT'] = np.nan
            missing_df['CLOUD_SCORE'] = np.nan
            # missing_df[f] = missing_df[f].astype(int)

            # Remove the overlapping missing entries
            # Then append the new missing entries
            if output_df.index.intersection(missing_df.index).any():
                output_df.drop(
                    output_df.index.intersection(missing_df.index),
                    inplace=True)
            output_df = output_df.append(missing_df)
            csv_writer(output_df, output_path, output_fields)

    # Identify SCENE_IDs that are missing any data
    # Filter based on product and SCENE_ID lists
    # Check for missing data as long as PIXEL_COUNT > 0
    missing_fields = landsat_fields[:]
    missing_id_mask = (
        (output_df['PIXEL_COUNT'] > 0) &
        output_df.index.isin(export_ids))
    missing_df = output_df.loc[missing_id_mask, missing_fields].isnull()

    # List of SCENE_IDs and products with some missing data
    missing_any_ids = set(missing_df[missing_df.any(axis=1)].index.values)

    # DEADBEEF - For now, skip SCENE_IDs that are only missing Ts
    if not missing_df.empty and 'TS' in missing_fields:
        missing_ts_ids = set(missing_df[
            missing_df[['TS']].any(axis=1) &
            ~missing_df.drop('TS', axis=1).any(axis=1)].index.values)
        if missing_ts_ids:
            logging.info('  SCENE_IDs missing Ts only: {}'.format(
                ', '.join(sorted(missing_ts_ids))))
            missing_any_ids -= missing_ts_ids
            # input('ENTER')

    # logging.debug('  SCENE_IDs missing all values: {}'.format(
    #     ', '.join(sorted(missing_all_ids))))
    # logging.debug('  SCENE_IDs missing any values: {}'.format(
    #     ', '.join(sorted(missing_any_ids))))

    # Check for fields that are entirely empty or not present
    #   These may have been added but not filled
    # Additional logic is to handle condition where
    #   calling all on an empty dataframe returns True
    if not missing_df.empty:
        missing_all_products = set(
            f.lower()
            for f in missing_df.columns[missing_df.all(axis=0)])
        missing_any_products = set(
            f.lower()
            for f in missing_df.columns[missing_df.any(axis=0)])
    else:
        missing_all_products = set()
        missing_any_products = set()
    if missing_all_products:
        logging.debug('    Products missing all values: {}'.format(
            ', '.join(sorted(missing_all_products))))
    if missing_any_products:
        logging.debug('    Products missing any values: {}'.format(
            ', '.join(sorted(missing_any_products))))

    missing_ids = missing_all_ids | missing_any_ids
    missing_products = missing_all_products | missing_any_products

    # If mosaic flag is set, switch IDs back to non-mosaiced
    if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
        missing_scene_ids = [
            scene_id for mosaic_id in missing_ids
            for scene_id in mosaic_id_dict[mosaic_id]]
    else:
        missing_scene_ids = set(missing_ids)
    # logging.debug('  SCENE_IDs missing: {}'.format(
    #     ', '.join(sorted(missing_scene_ids))))
    logging.info('    Missing ID count: {}'.format(
        len(missing_scene_ids)))

    # Evaluate whether a subset of SCENE_IDs or products can be exported
    # The SCENE_ID skip and keep lists cannot be mosaiced SCENE_IDs
    if not missing_scene_ids and not missing_products:
        logging.info('    No missing data or products, skipping zone')
        return True
    elif missing_scene_ids or missing_all_products:
        logging.info('    Exporting all products for specific SCENE_IDs')
        landsat.scene_id_keep_list = sorted(list(missing_scene_ids))
        landsat.products = landsat_products[:]
    elif missing_scene_ids and missing_products:
        logging.info('    Exporting specific missing products/SCENE_IDs')
        landsat.scene_id_keep_list = sorted(list(missing_scene_ids))
        landsat.products = list(missing_products)
    elif not missing_scene_ids and missing_products:
        # This conditional will happen when images are missing Ts only
        # The SCENE_IDs are skipped but the missing products is not being
        #   updated also.
        logging.info(
            '    Missing products but no missing SCENE_IDs, skipping zone')
        return True
    else:
        logging.error('    Unhandled conditional')
        input('ENTER')


    # Reset the Landsat collection args
    # Use SCENE_ID list to set Landsat type and tile filters
    landsat.set_landsat_from_scene_id()
    landsat.set_tiles_from_scene_id()
    # landsat.set_landsat_from_flags()
    # landsat.path_keep_list = landsat_args['path_keep_list']
    # landsat.row_keep_list = landsat_args['row_keep_list']
    landsat.mosaic_method = landsat_args['mosaic_method']
    # Originally this was set to None to capture all scenes
    #   including the non-intersecting ones
    landsat.zone_geom = None
    # landsat.zone_geom s= zone['geom']
    # pp.pprint(vars(landsat))
    # input('ENTER')

    def export_update(data_df):
        """Set/modify ancillary field values in the export CSV dataframe"""
        # First remove any extra rows that were added for exporting
        data_df.drop(
            data_df[data_df.SCENE_ID == 'DEADBEEF'].index, inplace=True)

        # # With old Fmask data, PIXEL_COUNT can be > 0 even if all data is NaN
        # if ('NDVI_TOA' in data_df.columns.values and
        #         'TS' in data_df.columns.values):
        #     drop_mask = (
        #         data_df['NDVI_TOA'].isnull() & data_df['TS'].isnull() &
        #         (data_df['PIXEL_COUNT'] > 0))
        #     if not data_df.empty and drop_mask.any():
        #         data_df.loc[drop_mask, ['PIXEL_COUNT']] = 0

        # Add additional fields to the export data frame
        data_df.set_index('SCENE_ID', inplace=True, drop=True)
        if not data_df.empty:
            # data_df['ZONE_NAME'] = data_df['ZONE_NAME'].astype(str)
            data_df['ZONE_FID'] = zone['fid']
            data_df['PLATFORM'] = data_df.index.str.slice(0, 4)
            data_df['PATH'] = data_df.index.str.slice(5, 8).astype(int)
            data_df['DATE'] = pd.to_datetime(
                data_df.index.str.slice(12, 20), format='%Y%m%d')
            data_df['YEAR'] = data_df['DATE'].dt.year
            data_df['MONTH'] = data_df['DATE'].dt.month
            data_df['DAY'] = data_df['DATE'].dt.day
            data_df['DOY'] = data_df['DATE'].dt.dayofyear.astype(int)
            data_df['AREA'] = zone['area']
            data_df['PIXEL_SIZE'] = landsat.cellsize

            fmask_mask = data_df['FMASK_TOTAL'] > 0
            if fmask_mask.any():
                data_df.loc[fmask_mask, 'FMASK_PCT'] = 100.0 * (
                    data_df.loc[fmask_mask, 'FMASK_COUNT'] /
                    data_df.loc[fmask_mask, 'FMASK_TOTAL'])
            data_df['QA'] = 0
            # data_fields = [
            #     p.upper()
            #     for p in landsat.products + ['CLOUD_SCORE', 'FMASK_PCT']]
            # data_df[data_fields] = data_df[data_fields].round(10)

        # Remove unused export fields
        if 'system:index' in data_df.columns.values:
            del data_df['system:index']
        if '.geo' in data_df.columns.values:
            del data_df['.geo']

        return data_df

    # Adjust start and end year to even multiples of year_step
    iter_start_year = ini['INPUTS']['start_year']
    iter_end_year = ini['INPUTS']['end_year'] + 1
    iter_years = ini['ZONAL_STATS']['year_step']
    if iter_years > 1:
        iter_start_year = int(math.floor(
            float(iter_start_year) / iter_years) * iter_years)
        iter_end_year = int(math.ceil(
            float(iter_end_year) / iter_years) * iter_years)

    # Process date range by year
    for year in range(iter_start_year, iter_end_year, iter_years):
        start_dt = datetime.datetime(year, 1, 1)
        end_dt = (
            datetime.datetime(year + iter_years, 1, 1) -
            datetime.timedelta(0, 1))
        start_date = start_dt.date().isoformat()
        end_date = end_dt.date().isoformat()
        start_year = max(start_dt.date().year, ini['INPUTS']['start_year'])
        end_year = min(end_dt.date().year, ini['INPUTS']['end_year'])

        # Skip year range if SCENE_ID keep list is set and doesn't match
        if (landsat.scene_id_keep_list and not any(
                set(int(x[12:16]) for x in landsat.scene_id_keep_list) &
                set(range(start_year, end_year + 1)))):
            logging.debug('  {}  {}'.format(start_date, end_date))
            logging.debug('    No matching SCENE_IDs for year range')
            continue
        else:
            logging.info('  {}  {}'.format(start_date, end_date))

        if iter_years > 1:
            year_str = '{}_{}'.format(start_year, end_year)
        else:
            year_str = '{}'.format(year)
        # logging.debug('  {}  {}'.format(start_year, end_year))

        # Include EPSG code in export and output names
        if 'EPSG' in ini['SPATIAL']['crs']:
            crs_str = '_' + ini['SPATIAL']['crs'].replace(':', '').lower()
        else:
            crs_str = ''

        # Export Landsat zonal stats
        export_id = '{}_{}_landsat{}_{}'.format(
            ini['INPUTS']['zone_filename'], zone['name'].lower(),
            crs_str, year_str)
        # export_id = '{}_{}_landsat_{}'.format(
        #     os.path.splitext(ini['INPUTS']['zone_filename'])[0],
        #     zone_name, year_str)
        # output_id = '{}_landsat{}_{}'.format(
        #     zone['name'], crs_str, year_str)
        if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
            export_id += '_' + ini['INPUTS']['mosaic_method'].lower()
            # output_id += '_' + ini['INPUTS']['mosaic_method'].lower()

        # Filter by iteration date in addition to input date parameters
        landsat.start_date = start_date
        landsat.end_date = end_date
        landsat_coll = landsat.get_collection()

        # # DEBUG - Test that the Landsat collection is getting built
        # print(landsat_coll.aggregate_histogram('SCENE_ID').getInfo())
        # input('ENTER')
        # print('Bands: {}'.format(
        #     [x['id'] for x in ee.Image(landsat_coll.first()).getInfo()['bands']]))
        # print('SceneID: {}'.format(
        #     ee.Image(landsat_coll.first()).getInfo()['properties']['SCENE_ID']))
        # input('ENTER')
        # if ee.Image(landsat_coll.first()).getInfo() is None:
        #     logging.info('    No images, skipping')
        #     continue

        # Calculate values and statistics
        # Build function in loop to set water year ETo/PPT values
        def zonal_stats_func(image):
            """"""
            scene_id = ee.String(image.get('SCENE_ID'))
            date = ee.Date(image.get('system:time_start'))
            # doy = ee.Number(date.getRelative('day', 'year')).add(1)
            bands = len(landsat.products) + 3

            # Using zone['geom'] as the geomtry should make it
            #   unnecessary to clip also
            input_mean = ee.Image(image) \
                .select(landsat.products + ['cloud_score', 'row']) \
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=zone['geom'],
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False,
                    tileScale=1,
                    maxPixels=zone['max_pixels'] * bands)

            # Count unmasked Fmask pixels to get pixel count
            # Count Fmask > 1 to get Fmask count (0 is clear and 1 is water)
            fmask_img = ee.Image(image).select(['fmask'])
            input_count = ee.Image([
                    fmask_img.gte(0).unmask().rename(['pixel']),
                    fmask_img.gt(1).rename(['fmask'])]) \
                .reduceRegion(
                    reducer=ee.Reducer.sum().combine(
                        ee.Reducer.count(), '', True),
                    geometry=zone['geom'],
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False,
                    tileScale=1,
                    maxPixels=zone['max_pixels'] * 3)

            # Standard output
            zs_dict = {
                'ZONE_NAME': str(zone['name']),
                # 'ZONE_FID': zone['fid'],
                'SCENE_ID': scene_id.slice(0, 20),
                # 'PLATFORM': scene_id.slice(0, 4),
                # 'PATH': ee.Number(scene_id.slice(5, 8)),
                'ROW': ee.Number(input_mean.get('row')),
                # Compute dominant row
                # 'ROW': ee.Number(scene_id.slice(8, 11)),
                # 'DATE': date.format('YYYY-MM-dd'),
                # 'YEAR': date.get('year'),
                # 'MONTH': date.get('month'),
                # 'DAY': date.get('day'),
                # 'DOY': doy,
                # 'AREA': zone['area'],
                # 'PIXEL_SIZE': landsat.cellsize,
                'PIXEL_COUNT': input_count.get('pixel_sum'),
                'PIXEL_TOTAL': input_count.get('pixel_count'),
                'FMASK_COUNT': input_count.get('fmask_sum'),
                'FMASK_TOTAL': input_count.get('fmask_count'),
                # 'FMASK_PCT': ee.Number(input_count.get('fmask_sum')) \
                #     .divide(ee.Number(input_count.get('fmask_count'))) \
                #     .multiply(100),
                'CLOUD_SCORE': input_mean.get('cloud_score')
                # 'QA': ee.Number(0)
            }
            # Product specific output
            if landsat.products:
                zs_dict.update({
                    p.upper(): input_mean.get(p.lower())
                    for p in landsat.products
                })

            # Count the number of pixels with ET* == 0
            if 'etstar_mean' in landsat.products:
                etstar_count = ee.Image(image) \
                    .select(['etstar_mean'], ['etstar_count']) \
                    .lte(ini['BEAMER']['etstar_threshold']) \
                    .reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=zone['geom'],
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=ini['EXPORT']['transform'],
                        bestEffort=False,
                        tileScale=1,
                        maxPixels=zone['max_pixels'] * bands)
                zs_dict.update({
                    'ETSTAR_COUNT': etstar_count.get('etstar_count')})

            return ee.Feature(None, zs_dict)
        stats_coll = landsat_coll.map(zonal_stats_func, False)

        # # DEBUG - Test the function for a single image
        # stats_info = zonal_stats_func(
        #     ee.Image(landsat_coll.first())).getInfo()
        # pp.pprint(stats_info['properties'])
        # input('ENTER')

        # # DEBUG - Print the stats info to the screen
        # stats_info = stats_coll.getInfo()
        # for ftr in stats_info['features']:
        #     pp.pprint(ftr)
        # input('ENTER')
        # # return False

        # Add a dummy entry to the stats collection
        format_dict = {
            'ZONE_NAME': 'DEADBEEF',
            'SCENE_ID': 'DEADBEEF',
            'ROW': -9999,
            'PIXEL_COUNT': -9999,
            'PIXEL_TOTAL': -9999,
            'FMASK_COUNT': -9999,
            'FMASK_TOTAL': -9999,
            'CLOUD_SCORE': -9999,
        }
        if 'etstar_mean' in landsat.products:
            format_dict.update({'ETSTAR_COUNT': -9999})
        format_dict.update({p.upper(): -9999 for p in landsat.products})
        stats_coll = ee.FeatureCollection(ee.Feature(None, format_dict)) \
            .merge(stats_coll)

        # # DEBUG - Print the stats info to the screen
        # stats_info = stats_coll.getInfo()
        # for ftr in stats_info['features']:
        #     pp.pprint(ftr)
        # input('ENTER')

        logging.debug('    Requesting data')
        export_info = utils.ee_getinfo(stats_coll)['features']
        export_df = pd.DataFrame([f['properties'] for f in export_info])
        export_df = export_update(export_df)

        # Save data to main dataframe
        if not export_df.empty:
            logging.debug('    Processing data')
            if overwrite_flag:
                # Update happens inplace automatically
                # output_df.update(export_df)
                output_df = output_df.append(export_df)
            else:
                # Combine first doesn't have an inplace parameter
                output_df = output_df.combine_first(export_df)
            # output_df.sort_values(by=['DATE', 'ROW'], inplace=True)
        # print(output_df.head(10))
        # input('ENTER')

    # Save updated CSV
    if output_df is not None and not output_df.empty:
        logging.info('    Writing CSV')
        # csv_writer(output_df, output_path, output_fields)
    else:
        logging.info(
            '  Empty output dataframe\n'
            '  The exported CSV files may not be ready')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine zonal statistics by zone',
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
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, overwrite_flag=args.overwrite)
