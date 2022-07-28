#--------------------------------
# Name:         ee_zonal_stats_by_image_gsheet.py
# Purpose:      Download zonal stats by image using Earth Engine
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
from collections import defaultdict
import datetime
from itertools import groupby
import json
import logging
import os
import pprint
import re
from subprocess import check_output
import sys

import ee
import gspread
import numpy as np
from oauth2client import service_account
from osgeo import ogr
import pandas as pd

# This is an awful way of getting the parent folder into the path
# We really should package this up as a module with a setup.py
# This way the ee_tools folders would be in the
#   PYTHONPATH env. variable
# ee_tools_path = os.path.dirname(os.path.dirname(
#     os.path.abspath(os.path.realpath(__file__))))
# sys.path.insert(0, os.path.join(ee_tools_path, 'ee_tools'))
# sys.path.insert(0, ee_tools_path)
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
    logging.info('\nEarth Engine zonal statistics by image')

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

    # # Zonal stats init file paths
    # zone_geojson_path = os.path.join(
    #     ini['ZONAL_STATS']['output_ws'],
    #     os.path.basename(ini['INPUTS']['zone_shp_path']).replace(
    #         '.shp', '.geojson'))

    # These may eventually be set in the INI file
    landsat_daily_fields = [
        'ZONE_NAME', 'DATE', 'SCENE_ID', 'PLATFORM',
        # 'ZONE_NAME', 'ZONE_FID', 'DATE', 'SCENE_ID', 'PLATFORM',
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
            'tc_green', 'tc_bright', 'tc_wet'])
        ini['ZONAL_STATS']['landsat_products'].remove('tasseled_cap')
    landsat_daily_fields.extend(
        [p.upper() for p in ini['ZONAL_STATS']['landsat_products']])

    # Convert the shapefile to geojson
    if not os.path.isfile(ini['ZONAL_STATS']['zone_geojson']) or overwrite_flag:
        logging.info('\nConverting zone shapefile to GeoJSON')
        logging.debug('  {}'.format(ini['ZONAL_STATS']['zone_geojson']))
        check_output([
            'ogr2ogr', '-f', 'GeoJSON', '-preserve_fid',
            '-select', '{}'.format(ini['INPUTS']['zone_field']),
            # '-lco', 'COORDINATE_PRECISION=2'
            ini['ZONAL_STATS']['zone_geojson'], ini['INPUTS']['zone_shp_path']])

    # # Get ee features from shapefile
    # zone_geom_list = gdc.shapefile_2_geom_list_func(
    #     ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
    #     reverse_flag=False)
    # # zone_count = len(zone_geom_list)
    # # output_fmt = '_{0:0%sd}.csv' % str(int(math.log10(zone_count)) + 1)

    # Read in the zone geojson
    logging.debug('\nReading zone GeoJSON')
    try:
        with open(ini['ZONAL_STATS']['zone_geojson'], 'r') as f:
            zones_geojson = json.load(f)
    except Exception as e:
        logging.error('  Error reading zone geojson file, removing')
        logging.debug('  Exception: {}'.format(e))
        os.remove(ini['ZONAL_STATS']['zone_geojson'])

    # Check if the zone_names are unique
    # Eventually support merging common zone_names
    zone_names = [
        str(z['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_').lower()
        for z in zones_geojson['features']]
    if len(set(zone_names)) != len(zones_geojson['features']):
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
    zones_osr = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zones_wkt = gdc.osr_wkt(zones_osr)

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zones_osr, ini['SPATIAL']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zones_osr))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.warning('  Zone Proj4:   {}'.format(
            zones_osr.ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['SPATIAL']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(zones_wkt))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['SPATIAL']['cellsize']))

    # Initialize Earth Engine API key
    logging.info('\nInitializing Earth Engine')
    ee.Initialize()
    utils.ee_getinfo(ee.Number(1))

    # Build separate tile lists for each zone
    # Build tile lists before filtering by FID below
    # DEADBEEF - This is a list of all "possible" tile that is
    #   independent of the INI tile settings
    ini['ZONAL_STATS']['zone_tile_json'] = {}
    ini['ZONAL_STATS']['tile_scene_json'] = {}
    if (os.path.isfile(ini['ZONAL_STATS']['zone_tile_path']) and
            not overwrite_flag):
        logging.debug('\nReading zone tile lists\n  {}'.format(
            ini['ZONAL_STATS']['zone_tile_path']))
        with open(ini['ZONAL_STATS']['zone_tile_path'], 'r') as f:
            ini['ZONAL_STATS']['zone_tile_json'] = json.load(f)
    else:
        logging.info('\nBuilding zone tile lists')
        step = 1000
        zone_n = len(zones_geojson['features'])
        for zone_i in range(0, len(zones_geojson['features']), step):
            logging.debug('  Zones: {}-{}'.format(
                zone_i, min(zone_i + step, zone_n) - 1))
            zone_ftr_sub = zones_geojson['features'][zone_i: min(zone_i + step, zone_n)]

            # Build the zones feature collection in a list comprehension
            #   in order to set the correct spatial reference
            zone_field = ini['INPUTS']['zone_field']
            zone_coll = ee.FeatureCollection([
                ee.Feature(
                    ee.Geometry(f['geometry'], zones_wkt, False),
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
        zones_geojson['features'] = [
            ftr for ftr in zones_geojson['features']
            if ftr['id'] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zones_geojson['features'] = [
            ftr for ftr in zones_geojson['features']
            if ftr['id'] not in ini['INPUTS']['fid_skip_list']]

    # Merge geometries (after filtering by FID above)
    if ini['INPUTS']['merge_geom_flag']:
        logging.debug('\nMerging geometries')
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone_ftr in zones_geojson['features']:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone_ftr['geometry'])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        zones_geojson['features'] = [{
            'type': 'Feature',
            'id': 0,
            'properties': {ini['INPUTS']['zone_field']: zones_geojson['name']},
            'geometry': json.loads(merge_geom.ExportToJson())}]

        # Collapse WRS2 tile lists for merged geometry
        ini['ZONAL_STATS']['zone_tile_json'][zones_geojson['name']] = sorted(list(set([
            pr for pr_list in ini['ZONAL_STATS']['zone_tile_json'].values()
            for pr in pr_list])))
        logging.debug('  WRS2 Tiles: {}'.format(
            ini['ZONAL_STATS']['zone_tile_json'][zones_geojson['name']]))

    # Get end date of GRIDMET (if needed)
    # This could be moved to inside the INI function
    if ini['ZONAL_STATS']['gridmet_monthly_flag']:
        gridmet_end_dt = utils.ee_getinfo(ee.Date(ee.Image(
            ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                .filterDate(
                    '{}-01-01'.format(ini['INPUTS']['end_year'] - 1),
                    '{}-01-01'.format(ini['INPUTS']['end_year'] + 1)) \
                .limit(1, 'system:time_start', False) \
                .first()
            ).get('system:time_start')).format('YYYY-MM-dd'))
        gridmet_end_dt = datetime.datetime.strptime(
            gridmet_end_dt, '%Y-%m-%d')
        logging.debug('\nLast GRIDMET date: {}'.format(gridmet_end_dt))

    # Calculate zonal stats for each image separately
    logging.debug('\nComputing zonal stats')
    if ini['ZONAL_STATS']['landsat_flag']:
        landsat_func(
            landsat_daily_fields, ini, zones_geojson, zones_wkt,
            overwrite_flag)


def reAuthSheet(sheet_id, sheet_name):
    # Authenticate each time?
    credentials = service_account.ServiceAccountCredentials.from_json_keyfile_name(
        CLIENT_SECRET_FILE, SCOPES)
    gc = gspread.authorize(credentials)
    return gc.open_by_key(sheet_id).worksheet(sheet_name)


def gsheet_writer(output_df, fields, sheet_id, sheet_name, n=20):
    """Write (append) dataframe to Google Sheet

    If function is not defined here, gsheet and fields will need to be
    passed into the function.

    """
    temp_df = output_df.copy()
    temp_df.reset_index(drop=False, inplace=True)
    temp_df.sort_values(['SCENE_ID', 'ZONE_NAME'], inplace=True)
    # temp_df.sort_values(['DATE', 'ZONE_NAME'], inplace=True)

    # Authenticate each time?
    gsheet = reAuthSheet(sheet_id, sheet_name)

    try:
        sheet_rows = gsheet.row_count
        logging.debug('  Sheet rows: {}'.format(sheet_rows))
    except Exception as e:
        logging.exception('\nException: {}\n  Skipping all rows'.format(e))
        input('ENTER')
        return False

    logging.debug('  Adding {} rows'.format(len(temp_df)))
    try:
        gsheet.add_rows(len(temp_df))
    except Exception as e:
        logging.warning('  Exception: {}\n  Skipping all rows'.format(e))
        input('ENTER')
        return False

    # Enumerate won't work for computing df_i when n > 1
    for df_i, row_i in zip(
            range(0, len(temp_df), n),
            range(sheet_rows + 1, sheet_rows + len(temp_df) + 1, n)):
        rows_df = temp_df.iloc[df_i: df_i + n]
        logging.info('  {}'.format(', '.join(rows_df['ZONE_NAME'])))

        cell_range = '{ul}:{lr}'.format(
            ul=gspread.utils.rowcol_to_a1(row_i, 1),
            lr=gspread.utils.rowcol_to_a1(
                row_i + (len(rows_df) - 1), len(fields)))
        logging.debug('  Cell Range: {}'.format(cell_range))
        try:
            cell_list = gsheet.range(cell_range)
        except Exception as e:
            logging.warning('  Exception: {}\n  Skipping rows'.format(e))
            continue

        start_row = cell_list[0].row
        for cell in cell_list:
            cell_i = cell.row - start_row
            value = rows_df.loc[rows_df.index[cell_i], fields[cell.col - 1]]
            # logging.debug('  {} {} {} {} {}'.format(
            #     cell_i, rows_df.index[cell_i], fields[cell.col - 1],
            #     value, type(value)))

            # Don't save nan values to sheet
            if isinstance(value, np.float64) and np.isnan(value):
                pass
            else:
                cell.value = value

        try:
            gsheet.update_cells(cell_list)
        except Exception as e:
            logging.warning('  Exception: {}\n  Skipping rows'.format(e))
            # logging.warning('  {}'.format(
            #     rows_df.loc[rows_df.index[df_i: df_i + new_rows - 1]]))
            continue


# def gsheet_writer(output_df):
#     """Write (append) dataframe to Google Sheet"""
#     temp_df = output_df.copy()
#     temp_df.reset_index(drop=False, inplace=True)

#     output_rows = len(temp_df)
#     logging.debug('  Adding {} rows'.format(output_rows))
#     rows = gsheet.row_count
#     gsheet.add_rows(output_rows)

#     cell_range = '{ul}:{lr}'.format(
#         ul=gspread.utils.rowcol_to_a1(rows + 1, 1),
#         lr=gspread.utils.rowcol_to_a1(
#             rows + output_rows, len(output_fields)))
#     logging.info('  Cell Range: {}'.format(cell_range))
#     cell_list = gsheet.range(cell_range)
#     for cell in cell_list:
#         value = temp_df.loc[
#             temp_df.index[cell.row - rows - 1], output_fields[cell.col - 1]]
#         cell.value = value
#     try:
#         gsheet.update_cells(cell_list)
#     except Exception as e:
#         logging.warning('  Exception: {}\n  Skipping'.format(e))
#         logging.warning('  {}'.format(cell_list))
#         input('ENTER')


def landsat_func(export_fields, ini, zones_geojson, zones_wkt,
                 overwrite_flag=False):
    """

    Function will attempt to generate export tasks only for missing SCENE_IDs
    Also try to limit the products to only those with missing data

    Parameters
    ----------
    export_fields :
    ini : dict
        Input file parameters.
    zones_geojson (dict):
        Zones GeoJSON.
    zones_wkt :
        Zones spatial reference Well Known Text.
    overwrite_flag : bool, optional
        If True, overwrite existing values (the default is False).
        Don't remove/replace the CSV file directly.

    """
    # logging.info('\nLandsat')

    # DEADBEEF - For now, hardcode transform to a standard Landsat image
    ini['EXPORT']['transform'] = [30.0, 0.0, 15.0, 0.0, -30.0, 15.0]
    # ini['EXPORT']['transform'] = '[{}]'.format(','.join(
    #     map(str, 30.0, 0.0, 15.0, 0.0, -30.0, 15.0)))
    # logging.debug('  Output Transform: {}'.format(
    #     ini['EXPORT']['transform']))

    landsat_products = ini['ZONAL_STATS']['landsat_products'][:]
    # landsat_fields = [f.upper() for f in landsat_products]
    # logging.debug('  Products: {}'.format(', '.join(landsat_products)))
    # logging.debug('  Fields:   {}'.format(', '.join(landsat_fields)))

    # Assuming Google Sheet exists and has the target columns
    # Assuming worksheet is called "Landsat_Daily"
    logging.info('\nReading Landsat GSHEET')
    gsheet = reAuthSheet(
        ini['GSHEET']['gsheet_id'], ini['GSHEET']['landsat_daily'])
    try:
        output_fields = gsheet.row_values(1)
        logging.debug('  Sheet fields: {}'.format(', '.join(output_fields)))
    except Exception as e:
        logging.exception('\nException: {}'.format(e))

    # # DEADBEEF - Changing the field structure should probably be done manually
    # # Add missing fields
    # for field_name in export_fields:
    #     if field_name not in output_fields:
    #         logging.info('  Adding field: {}'.format(field_name))
    #         gsheet.add_cols(1)
    #         gsheet.update_cell(1, gsheet.col_count, field_name)
    #         output_fields.append(field_name)

    # Read all data from Google Sheet
    logging.info('Retrieving all values')
    input_data = gsheet.get_all_values()

    # Remove empty rows after last row with data
    logging.info('Removing empty rows from Google Sheet')
    gsheet_rows = gsheet.row_count
    data_rows = len(input_data)
    logging.debug('  Sheet rows: {}'.format(gsheet_rows))
    logging.debug('  Data rows:  {}'.format(data_rows))
    gsheet.resize(rows=data_rows)
    # for row_i in range(gsheet.row_count, data_rows, -1):
    #     # logging.debug('  Deleting row: {}'.format(row_i))
    #     gsheet.delete_row(row_i)

    # Remove empty rows in data (gsheet row indexing is 1 based)
    for row_i in range(len(input_data), 1, -1):
        if not any(input_data[row_i - 1]):
            logging.debug('  Deleting row: {}'.format(row_i))
            gsheet.delete_row(row_i)
            del input_data[row_i - 1]

    # Build input/output dataframes
    input_df = pd.DataFrame(data=input_data[1:], columns=output_fields)
    output_df = pd.DataFrame(data=None, columns=input_df.columns, index=None)
    input_df.set_index(['SCENE_ID', 'ZONE_NAME'], inplace=True, drop=True)
    output_df.set_index(['SCENE_ID', 'ZONE_NAME'], inplace=True, drop=True)
    # print(input_df)
    # print(output_df)
    # input('ENTER')

    # Master list of zone dictionaries (name, fid, csv and ee.Geometry)
    zones = []
    for z_ftr in zones_geojson['features']:
        zone_name = str(
            z_ftr['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_')

        # zone_geom = ogr.CreateGeometryFromJson(json.dumps(zone['json']))
        # # zone_area = zone_geom.GetArea()
        # if zone_geom.GetGeometryName() in ['POINT', 'MULTIPOINT']:
        #     # Compute area as cellsize * number of points
        #     point_count = 0
        #     for i in range(0, zone_geom.GetGeometryCount()):
        #         point_count += zone_geom.GetGeometryRef(i).GetPointCount()
        #     zone['area'] = ini['SPATIAL']['cellsize'] * point_count
        # else:
        #     # Adjusting area up to nearest multiple of cellsize to account for
        #     #   polygons that were modified to avoid interior holes
        #     zone_area = ini['SPATIAL']['cellsize'] * math.ceil(
        #         zone_geom.GetArea() / ini['SPATIAL']['cellsize'])

        # I have to build an EE geometry in order to set a non-WGS84 projection
        zones.append({
            'name': zone_name,
            'fid': int(z_ftr['id']),
            # 'area': zone_geom.GetArea(),
            'csv': os.path.join(
                ini['ZONAL_STATS']['output_ws'], zone_name,
                '{}_landsat_daily.csv'.format(zone_name)),
            'ee_geom': ee.Geometry(
                geo_json=z_ftr['geometry'], opt_proj=zones_wkt,
                opt_geodesic=False)
        })

    # Master list of tiles to process
    logging.debug('\nFiltering zone tiles')
    output_tiles = set()
    for zone in zones:
        logging.debug('  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))

        # Pre-filter by tile
        # First get the list of possible tiles for each zone
        try:
            zone_tile_list = ini['ZONAL_STATS']['zone_tile_json'][zone['name']]
        except KeyError:
            logging.debug('    No matching tiles, skipping zone')
            continue

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
            logging.debug('    No matching tiles, skipping zone')
            continue

        output_tiles.update(zone_tile_list)

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
    landsat.zone_geom = None
    landsat.products = []
    landsat.mosaic_method = 'none'

    # Build a master list of available SCENE_IDs for each path/row
    #   based on the INI parameters (set above)
    # The tile_scene_json dictionary starts off empty but is populated while
    #   iterating through the tiles.
    output_ids = set()
    logging.debug('\n  Building tiles scene lists')
    for tile in output_tiles:
        logging.debug('  {}'.format(tile))
        if tile in ini['ZONAL_STATS']['tile_scene_json']:
            # Read the previously computed tile SCENE_ID list
            output_ids.update(
                ini['ZONAL_STATS']['tile_scene_json'][tile])
        else:
            # Compute the SCENE_ID list for each tile if needed
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
            ini['ZONAL_STATS']['tile_scene_json'][tile] = list(utils.ee_getinfo(
                landsat_coll.aggregate_histogram('SCENE_ID')).keys())
            output_ids.update(ini['ZONAL_STATS']['tile_scene_json'][tile])
        logging.debug('  {}'.format(
            ', '.join(ini['ZONAL_STATS']['tile_scene_json'][tile])))

    # If export_ids is empty, all SCENE_IDs may have been filtered
    if not output_ids:
        logging.info('\n  No SCENE_IDs to process after applying INI filters')
        return False

    # Compute mosaiced SCENE_IDs after filtering
    if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
        mosaic_id_dict = defaultdict(list)
        for scene_id in output_ids:
            mosaic_id = '{}XXX{}'.format(scene_id[:8], scene_id[11:])
            mosaic_id_dict[mosaic_id].append(scene_id)
        export_ids = set(mosaic_id_dict.keys())

    # # For overwrite, drop all expected entries from existing output DF
    # if overwrite_flag:
    #     output_df = output_df[
    #         ~output_df.index.get_level_values('SCENE_ID').isin(list(export_ids))]
    #     # DEADBEEF - This doesn't work for some rearson
    #     # output_df.drop(
    #     #     output_df.index.get_level_values('SCENE_ID').isin(list(export_ids)),
    #     #     inplace=True)

    # zone_df below is intentionally being made as a subset copy of output_df
    # This raises a warning though
    pd.options.mode.chained_assignment = None  # default='warn'

    # Add empty entries separately for each zone
    logging.debug('\n  Identifying scenes/zones with missing data')
    for zone in zones:
        logging.debug('  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))

        # Compute custom SCENE_ID lists for each zone
        # This could probably simplified or moved out of the loop
        export_ids = set()
        for zone_tile in ini['ZONAL_STATS']['zone_tile_json'][zone['name']]:
            if zone_tile in ini['ZONAL_STATS']['tile_scene_json'].keys():
                for id in ini['ZONAL_STATS']['tile_scene_json'][zone_tile]:
                    if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
                        export_ids.add('{}XXX{}'.format(id[:8], id[11:]))
                    else:
                        export_ids.add(id)
        if not export_ids:
            logging.info('    No SCENE_IDs to process, skipping zone')
            continue

        # Subset the data frame
        zone_df = input_df.iloc[
            input_df.index.get_level_values('ZONE_NAME')==zone['name']]
        zone_df.reset_index(inplace=True)
        zone_df.set_index(['SCENE_ID'], inplace=True, drop=True)

        # Get list of existing dates in the sheet
        if not zone_df.empty:
            zone_df_ids = set(zone_df.index.values)
        else:
            zone_df_ids = set()
        logging.debug('    Scenes in sheet: {}'.format(
            ', '.join(zone_df_ids)))

        # List of SCENE_IDs that are entirely missing
        # This may include scenes that don't intersect the zone
        missing_all_ids = export_ids - zone_df_ids
        # logging.debug('    Scenes not in CSV (missing all values): {}'.format(
        #     ', '.join(sorted(missing_all_ids))))
        if not missing_all_ids:
            continue

        # # Get a list of SCENE_IDs that intersect the zone
        # # Get SCENE_ID list mimicking a full extract below
        # #   (but without products)
        # # Start with INI path/row keep list but update based on SCENE_ID later
        # landsat.products = []
        # landsat.path_keep_list = landsat_args['path_keep_list']
        # landsat.row_keep_list = landsat_args['row_keep_list']
        # landsat.zone_geom = zone['ee_geom']
        # landsat.mosaic_method = landsat_args['mosaic_method']
        #
        # # Only use the scene_id_keep_list if there was already data in the DF
        # if zone_df.index.values.any():
        #     # SCENE_ID keep list must be non-mosaiced IDs
        #     if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
        #         landsat.scene_id_keep_list = sorted(set([
        #             scene_id for mosaic_id in missing_all_ids
        #             for scene_id in mosaic_id_dict[mosaic_id]]))
        #     else:
        #         landsat.scene_id_keep_list = sorted(missing_all_ids)
        #     landsat.set_landsat_from_scene_id()
        #     landsat.set_tiles_from_scene_id()
        #
        # # Get the SCENE_IDs that intersect the zone
        # # Process each Landsat type independently
        # # Was having issues getting the full scene list at once
        # logging.debug('    Getting intersecting SCENE_IDs')
        # missing_zone_ids = set()
        # landsat_type_list = landsat._landsat_list[:]
        # for landsat_str in landsat_type_list:
        #     landsat._landsat_list = [landsat_str]
        #     missing_zone_ids.update(set(utils.ee_getinfo(
        #         landsat.get_collection().aggregate_histogram('SCENE_ID'))))
        #     # logging.debug('      {} {}'.format(
        #     #     landsat_str, len(missing_zone_ids)))
        # landsat._landsat_list = landsat_type_list
        #
        # # # Get the SCENE_IDs that intersect the zone
        # # logging.debug('    Getting intersecting SCENE_IDs')
        # missing_zone_ids = set(utils.ee_getinfo(
        #     landsat.get_collection().aggregate_histogram('SCENE_ID')))
        #
        # # Difference of sets are SCENE_IDs that don't intersect
        # missing_nonzone_ids = missing_all_ids - missing_zone_ids
        #
        # # Remove skipped/empty SCENE_IDs from possible SCENE_ID list
        # # DEADBEEF - This is modifying the master export_id list
        # #   Commenting out for now
        # # export_ids = export_ids - missing_skip_ids
        # logging.debug('    Scenes intersecting zone: {}'.format(
        #     ', '.join(sorted(missing_zone_ids))))
        # logging.debug('    Scenes not intersecting zone: {}'.format(
        #     ', '.join(sorted(missing_nonzone_ids))))
        # logging.info('    Include ID count: {}'.format(
        #     len(missing_zone_ids)))
        # logging.info('    Exclude ID count: {}'.format(
        #     len(missing_nonzone_ids)))

        # Create empty entries for all scenes that are missing
        # Identify whether the missing scenes intersect the zone or not
        # Set pixels counts for non-intersecting SCENE_IDs to 0
        # Set pixel counts for intersecting SCENE_IDs to NaN
        #   This will allow the zonal stats to set actual pixel count value
        # logging.debug('    Appending all empty SCENE_IDs')
        missing_df = pd.DataFrame(
            index=missing_all_ids, columns=output_df.columns)
        missing_df.index.name = 'SCENE_ID'
        missing_df['ZONE_NAME'] = zone['name']
        # missing_df['ZONE_FID'] = zone['fid']
        # missing_df['AREA'] = zone['area']
        missing_df['PLATFORM'] = missing_df.index.str.slice(0, 4)
        missing_df['PATH'] = missing_df.index.str.slice(5, 8).astype(int)
        missing_df['DATE'] = pd.to_datetime(
            missing_df.index.str.slice(12, 20), format='%Y%m%d')
        missing_df['YEAR'] = missing_df['DATE'].dt.year
        missing_df['MONTH'] = missing_df['DATE'].dt.month
        missing_df['DAY'] = missing_df['DATE'].dt.day
        missing_df['DOY'] = missing_df['DATE'].dt.dayofyear.astype(int)
        missing_df['DATE'] = missing_df['DATE'].dt.strftime('%Y-%m-%d')
        # DEADBEEF - Setting ROW to np.nan will cause problems in QAQC and
        #   summary tables unless fixed later on
        # ROW needs to be set to NAN so that update call will replace it
        missing_df['ROW'] = np.nan
        missing_df['QA'] = np.nan
        missing_df['PIXEL_SIZE'] = landsat.cellsize
        missing_df['PIXEL_COUNT'] = np.nan
        missing_df['PIXEL_TOTAL'] = np.nan
        missing_df['FMASK_COUNT'] = np.nan
        missing_df['FMASK_TOTAL'] = np.nan
        missing_df['FMASK_PCT'] = np.nan
        if 'etstar_mean' in landsat.products:
            missing_df['ETSTAR_COUNT'] = np.nan
        missing_df['CLOUD_SCORE'] = np.nan

        # Update the output dataframe
        missing_df.reset_index(inplace=True)
        missing_df.set_index(
            ['SCENE_ID', 'ZONE_NAME'], inplace=True, drop=True)
        output_df = output_df.append(missing_df)

        # # # Set intersecting zone pixel counts to 0
        # # if missing_nonzone_ids:
        # #     missing_nonzone_mask = missing_df.index.isin(missing_nonzone_ids)
        # #     missing_df.loc[missing_nonzone_mask, 'ROW'] = 0
        # #     missing_df.loc[missing_nonzone_mask, 'PIXEL_COUNT'] = 0
        # #     missing_df.loc[missing_nonzone_mask, 'PIXEL_TOTAL'] = 0
        # #     missing_df.loc[missing_nonzone_mask, 'FMASK_COUNT'] = 0
        # #     missing_df.loc[missing_nonzone_mask, 'FMASK_TOTAL'] = 0

        # # Remove the overlapping missing entries
        # # Then append the new missing entries to the zone CSV
        # # if zone_df.index.intersection(missing_df.index).any():
        # try:
        #     zone_df.drop(
        #         zone_df.index.intersection(missing_df.index),
        #         inplace=True)
        # except ValueError:
        #     pass
        # zone_df = zone_df.append(missing_df)

        # # Update the master dataframe
        # zone_df.reset_index(inplace=True)
        # zone_df.set_index(
        #     ['SCENE_ID', 'ZONE_NAME'], inplace=True, drop=True)
        # try:
        #     output_df.drop(
        #         output_df.index.get_level_values('ZONE_NAME')==zone['name'],
        #         inplace=True)
        # except Exception as e:
        #     # These seem to happen with the zone is not in the output_df
        #     logging.debug('    Exception: {}'.format(e))
        #     pass
        # output_df = output_df.append(zone_df)

    # Putting the warning back to the default balue
    pd.options.mode.chained_assignment = 'warn'

    # Identify SCENE_IDs that are missing any data
    # Filter based on product and SCENE_ID lists
    missing_id_mask = output_df.index.get_level_values('SCENE_ID') \
        .isin(export_ids)
    # missing_id_mask = (
    #     (output_df['PIXEL_COUNT'] != 0) &
    #     output_df.index.get_level_values('SCENE_ID').isin(export_ids))
    products = [f.upper() for f in ini['ZONAL_STATS']['landsat_products']]
    missing_df = output_df.loc[missing_id_mask, products].isnull()

    # List of SCENE_IDs and products with some missing data
    missing_ids = set(missing_df[missing_df.any(axis=1)]
        .index.get_level_values('SCENE_ID').values)

    # Skip processing if all dates already exist in the CSV
    if not missing_ids and not overwrite_flag:
        logging.info('\n  All scenes present, returning')
        return True
    else:
        logging.debug('\n  Scenes missing values: {}'.format(
            ', '.join(sorted(missing_ids))))

    # Reset the Landsat collection args
    landsat.path_keep_list = landsat_args['path_keep_list']
    landsat.row_keep_list = landsat_args['row_keep_list']
    landsat.products = ini['ZONAL_STATS']['landsat_products']
    landsat.mosaic_method = landsat_args['mosaic_method']

    def export_update(data_df):
        """Set/modify ancillary field values in the export CSV dataframe"""
        # First, remove any extra rows that were added for exporting
        # The DEADBEEF entry is added because the export structure is based on
        #   the first feature in the collection, so fields with nodata will
        #   be excluded
        data_df.drop(
            data_df[data_df['SCENE_ID'] == 'DEADBEEF'].index,
            inplace=True)

        # Add additional fields to the export data frame
        if not data_df.empty:
            data_df['PLATFORM'] = data_df['SCENE_ID'].str.slice(0, 4)
            data_df['PATH'] = data_df['SCENE_ID'].str.slice(5, 8).astype(int)
            data_df['DATE'] = pd.to_datetime(
                data_df['SCENE_ID'].str.slice(12, 20), format='%Y%m%d')
            data_df['YEAR'] = data_df['DATE'].dt.year
            data_df['MONTH'] = data_df['DATE'].dt.month
            data_df['DAY'] = data_df['DATE'].dt.day
            data_df['DOY'] = data_df['DATE'].dt.dayofyear.astype(int)
            data_df['DATE'] = data_df['DATE'].dt.strftime('%Y-%m-%d')
            data_df['AREA'] = data_df['PIXEL_COUNT'] * landsat.cellsize ** 2
            data_df['PIXEL_SIZE'] = landsat.cellsize

            fmask_mask = data_df['FMASK_TOTAL'] > 0
            if fmask_mask.any():
                data_df.loc[fmask_mask, 'FMASK_PCT'] = 100.0 * (
                    data_df.loc[fmask_mask, 'FMASK_COUNT'] /
                    data_df.loc[fmask_mask, 'FMASK_TOTAL'])
            data_df['QA'] = 0

        # Remove unused export fields
        if 'system:index' in data_df.columns.values:
            del data_df['system:index']
        if '.geo' in data_df.columns.values:
            del data_df['.geo']

        return data_df

    # Write values to file after each year
    # Group export dates by year (and path/row also?)
    export_ids_iter = [
        [year, list(dates)]
        for year, dates in groupby(sorted(missing_ids), lambda x: x[12:16])]
    for export_year, export_ids in export_ids_iter:
        logging.debug('\n  Iter year: {}'.format(export_year))

        # These can be mosaiced or single scene IDs depending on mosaic method
        for export_id in sorted(export_ids, key=lambda x: x[-8:]):
            logging.info('  {}'.format(export_id))

            if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
                export_id_list = mosaic_id_dict[export_id]
                logging.debug('    SCENE_IDs: {}'.format(
                    ', '.join(export_id_list)))
            else:
                export_id_list = list(export_id)

            # DEADBEEF - There has to be a way to do this selection in one line
            scene_df = output_df.iloc[
                output_df.index.get_level_values('SCENE_ID') == export_id]
            scene_df = scene_df[
                (scene_df['PIXEL_COUNT'] > 0) |
                (scene_df['PIXEL_COUNT'].isnull())]
            if scene_df.empty:
                logging.info('\n  No missing data, skipping')
                input('ENTER')
                continue
            scene_df.reset_index(inplace=True)
            scene_df.set_index(['ZONE_NAME'], inplace=True, drop=True)

            # Update/limit products list if necessary
            export_products = set(
                f.lower()
                for f in scene_df.isnull().any(axis=0).index.values
                if f.lower() in landsat_products)
            logging.debug('  Products missing any values: {}'.format(
                ', '.join(sorted(export_products))))
            landsat.products = list(export_products)

            # Identify zones with any missing data
            export_zones = set(scene_df[scene_df.any(axis=1)].index)
            logging.debug('  Zones with missing data: {}'.format(
                ', '.join(sorted(export_zones))))

            # Build collection of all features to test for each SCENE_ID
            zone_coll = ee.FeatureCollection([
                ee.Feature(
                    zone['ee_geom'],
                    {
                        'ZONE_NAME': zone['name'],
                        # 'ZONE_FID': zone['fid']
                    })
                for zone in zones if zone['name'] in export_zones])

            # Collection should only have one image
            landsat.scene_id_keep_list = export_id_list[:]
            # landsat.update_scene_id_keep(export_id_list)
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

            # Map over features for one image
            image = ee.Image(landsat_coll.first())

            def zonal_stats_func(ftr):
                """"""
                scene_id = ee.String(image.get('SCENE_ID'))
                # date = ee.Date(image.get('system:time_start'))
                # doy = ee.Number(date.getRelative('day', 'year')).add(1)
                # bands = len(landsat.products) + 3

                # Using the feature/geometry should make it unnecessary to clip
                input_mean = ee.Image(image) \
                    .select(landsat.products + ['cloud_score', 'row']) \
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=ftr.geometry(),
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=ini['EXPORT']['transform'],
                        bestEffort=False,
                        tileScale=1)
                        # maxPixels=zone['max_pixels'] * bands)

                # Count unmasked Fmask pixels to get pixel count
                # Count Fmask > 1 to get Fmask count (0 is clear and 1 is water)
                fmask_img = ee.Image(image).select(['fmask'])
                input_count = ee.Image([
                        fmask_img.gte(0).unmask().rename(['pixel']),
                        fmask_img.gt(1).rename(['fmask'])]) \
                    .reduceRegion(
                        reducer=ee.Reducer.sum().combine(
                            ee.Reducer.count(), '', True),
                        geometry=ftr.geometry(),
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=ini['EXPORT']['transform'],
                        bestEffort=False,
                        tileScale=1)
                        # maxPixels=zone['max_pixels'] * 3)

                # Standard output
                zs_dict = {
                    'ZONE_NAME': ee.String(ftr.get('ZONE_NAME')),
                    # 'ZONE_FID': ee.Number(ftr.get('ZONE_FID')),
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
                    # 'AREA': input_count.get('pixel_count') * (landsat.cellsize ** 2)
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
                            geometry=ftr.geometry(),
                            crs=ini['SPATIAL']['crs'],
                            crsTransform=ini['EXPORT']['transform'],
                            bestEffort=False,
                            tileScale=1)
                            # maxPixels=zone['max_pixels'] * bands)
                    zs_dict.update({
                        'ETSTAR_COUNT': etstar_count.get('etstar_count')})

                return ee.Feature(None, zs_dict)
            stats_coll = zone_coll.map(zonal_stats_func, False)

            # Add a dummy entry to the stats collection
            # This is added because the export structure is based on the first
            #   entry in the collection, so fields with nodata will be excluded
            format_dict = {
                'ZONE_NAME': 'DEADBEEF',
                # 'ZONE_FID': -9999,
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
            export_df = pd.DataFrame([
                ftr['properties']
                for ftr in utils.ee_getinfo(stats_coll)['features']])
            export_df = export_update(export_df)
            export_df.set_index(['ZONE_NAME'], inplace=True, drop=True)
            # export_df.set_index(
            #     ['SCENE_ID', 'ZONE_NAME'], inplace=True, drop=True)

            # Save data to main dataframe
            if not export_df.empty:
                logging.debug('    Processing data')
                if overwrite_flag:
                    # Update happens inplace automatically
                    scene_df.update(export_df)
                    # scene_df = scene_df.append(export_df)
                else:
                    # Combine first doesn't have an inplace parameter
                    scene_df = scene_df.combine_first(export_df)

            # Save data Google Sheet
            if not export_df.empty:
                logging.debug('    Saving data')
                gsheet_writer(
                    scene_df, output_fields, ini['GSHEET']['gsheet_id'],
                    ini['GSHEET']['landsat_daily'])

            # # Save data to main dataframe
            # if not export_df.empty:
            #     logging.debug('    Processing data')
            #     if overwrite_flag:
            #         # Update happens inplace automatically
            #         output_df.update(export_df)
            #         # output_df = output_df.append(export_df)
            #     else:
            #         # Combine first doesn't have an inplace parameter
            #         output_df = output_df.combine_first(export_df)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine zonal statistics by image',
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
