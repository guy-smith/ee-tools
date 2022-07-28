#--------------------------------
# Name:         ee_zonal_stats_by_image.py
# Purpose:      Export zonal stats to Google Sheet
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
import json
import logging
import os
import pprint
import sys
import time

import gspread
import numpy as np
from oauth2client import service_account
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
import ee_tools.inputs as inputs
import ee_tools.utils as utils

pp = pprint.PrettyPrinter(indent=4)

SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'MapWater-4e2df36b1209.json'


def main(ini_path=None, overwrite_flag=False):
    """Export zonal stats to Google Sheet

    Parameters
    ----------
    ini_path : str
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    """
    logging.info('\nExport zonal stats to Google Sheet')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    # inputs.parse_section(ini, section='SPATIAL')
    # inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='ZONAL_STATS')
    inputs.parse_section(ini, section='GSHEET')

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
    gridmet_daily_fields = [
        'ZONE_NAME', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'WATER_YEAR', 'ETO', 'PPT']
    gridmet_monthly_fields = [
        'ZONE_NAME', 'DATE', 'YEAR', 'MONTH', 'WATER_YEAR', 'ETO', 'PPT']

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

    # # Convert the shapefile to geojson
    # if not os.path.isfile(ini['ZONAL_STATS']['zone_geojson']) or overwrite_flag:
    #     logging.info('\nConverting zone shapefile to GeoJSON')
    #     logging.debug('  {}'.format(ini['ZONAL_STATS']['zone_geojson']))
    #     check_output([
    #         'ogr2ogr', '-f', 'GeoJSON', '-preserve_fid',
    #         '-select', '{}'.format(ini['INPUTS']['zone_field']),
    #         # '-lco', 'COORDINATE_PRECISION=2'
    #         ini['ZONAL_STATS']['zone_geojson'], ini['INPUTS']['zone_shp_path']])

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

    # Authenticate with Google Sheets API
    logging.debug('\nAuthenticating with Google Sheets API')
    oauth_cred = service_account.ServiceAccountCredentials.from_json_keyfile_name(
        CLIENT_SECRET_FILE, SCOPES)
    gsheet_cred = gspread.authorize(oauth_cred)

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

    # Calculate zonal stats for each image separately
    logging.debug('\nComputing zonal stats')
    if ini['ZONAL_STATS']['landsat_flag']:
        landsat_func(
            gsheet_cred, landsat_daily_fields, ini, zones_geojson,
            overwrite_flag)
    if ini['ZONAL_STATS']['gridmet_daily_flag']:
        gridmet_daily_func(
            gsheet_cred, gridmet_daily_fields, ini, zones_geojson,
            overwrite_flag)
    if ini['ZONAL_STATS']['gridmet_monthly_flag']:
        gridmet_monthly_func(
            gsheet_cred, gridmet_monthly_fields, ini, zones_geojson,
            overwrite_flag)


def reAuthSheet(sheet_id, sheet_name):
    # Authenticate each time?
    credentials = service_account.ServiceAccountCredentials.from_json_keyfile_name(
        CLIENT_SECRET_FILE, SCOPES)
    gc = gspread.authorize(credentials)
    return gc.open_by_key(sheet_id).worksheet(sheet_name)


def gsheet_writer(output_df, fields, sheet_id, sheet_name, n=100):
    """Write (append) dataframe to Google Sheet

    If function is not defined here, gsheet and fields will need to be
    passed into the function.

    """
    temp_df = output_df.copy()
    temp_df.reset_index(drop=False, inplace=True)
    # temp_df.sort_values(['SCENE_ID', 'ZONE_NAME'], inplace=True)
    temp_df.sort_values(['DATE', 'ZONE_NAME'], inplace=True)

    # Authenticate each time?
    gsheet = reAuthSheet(sheet_id, sheet_name)

    try:
        sheet_rows = gsheet.row_count
        logging.debug('  Sheet rows: {}'.format(sheet_rows))
    except Exception as e:
        logging.exception('\nException: {}\n  Skipping all rows'.format(e))
        input('ENTER')
        return False

    # # Add all export rows
    # logging.debug('  Adding {} rows'.format(len(temp_df)))
    # try:
    #     gsheet.add_rows(len(temp_df))
    # except Exception as e:
    #     logging.warning('  Exception: {}\n  Skipping all rows'.format(e))
    #     input('ENTER')
    #     return False

    # Enumerate won't work for computing df_i when n > 1
    row_i = sheet_rows + 1
    for df_i in range(0, len(temp_df), n):
        rows_df = temp_df.iloc[df_i: df_i + n]
        # logging.info('  {}'.format(', '.join(rows_df['ZONE_NAME'])))
        # logging.debug('  {} {}'.format(df_i, row_i))

        # Add blocks of empty rows
        logging.debug('  Adding {} rows'.format(len(rows_df)))
        try:
            gsheet.add_rows(len(rows_df))
        except Exception as e:
            logging.warning('  Exception: {}\n  Skipping rows'.format(e))
            input('ENTER')
            continue

        cell_range = '{ul}:{lr}'.format(
            ul=gspread.utils.rowcol_to_a1(row_i, 1),
            lr=gspread.utils.rowcol_to_a1(
                row_i + (len(rows_df) - 1), len(fields)))
        logging.debug('  Cell Range: {}'.format(cell_range))
        try:
            cell_list = gsheet.range(cell_range)
        except Exception as e:
            logging.warning('  Exception: {}\n  Skipping rows'.format(e))
            input('ENTER')
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
            elif isinstance(value, np.float64):
                cell.value = float(value)
            elif isinstance(value, np.int64):
                cell.value = int(value)
            elif fields[cell.col - 1] == 'DATE':
                # GSheet epoch is 1899-12-30
                cell.value = (
                    datetime.datetime.strptime(value, '%Y-%m-%d') -
                    datetime.datetime(1899, 12, 30)).days
            else:
                cell.value = value

        try:
            gsheet.update_cells(cell_list)
        except Exception as e:
            logging.warning('  Exception: {}\n  Skipping rows'.format(e))
            # logging.warning('  {}'.format(
            #     rows_df.loc[rows_df.index[df_i: df_i + new_rows - 1]]))
            input('ENTER')
            continue

        row_i += n


def landsat_func(gsheet_cred, export_fields, ini, zones_geojson,
                 overwrite_flag=False):
    """

    Function will attempt to generate export tasks only for missing SCENE_IDs
    Also try to limit the products to only those with missing data

    Parameters
    ----------
    gsheet_cred
    export_fields : list
    ini : dict
        Input file parameters
    zones_geojson : dict
        Zones GeoJSON
    overwrite_flag : bool
        If True, overwrite existing values (the default is False).
        Don't remove/replace the CSV file directly.

    """
    logging.info('\nLandsat')

    landsat_products = ini['ZONAL_STATS']['landsat_products'][:]
    landsat_fields = [f.upper() for f in landsat_products]

    # Assuming Google Sheet exists and has the target columns
    # Assuming worksheet is called "Landsat_Daily"
    logging.info('Reading Landsat GSHEET')
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

    # Read all data from Google Sheet
    logging.info('Retrieving all values')
    input_data = gsheet.get_all_values()

    # Remove empty rows after last row with data
    logging.info('Removing empty rows from Google Sheet')
    gsheet_rows = gsheet.row_count
    data_rows = len(input_data)
    logging.debug('  Sheet rows: {}'.format(gsheet_rows))
    logging.debug('  Data rows:  {}'.format(data_rows))
    for row_i in range(gsheet.row_count, data_rows, -1):
        # logging.debug('  Deleting row: {}'.format(row_i))
        gsheet.delete_row(row_i)

    # # Remove empty rows in data (gsheet row indexing is 1 based)
    # for row_i in range(len(input_data), 1, -1):
    #     if not any(input_data[row_i - 1]):
    #         logging.debug('  Deleting row: {}'.format(row_i))
    #         gsheet.delete_row(row_i)
    #         del input_data[row_i - 1]

    # Build sheet dataframe
    sheet_df = pd.DataFrame(data=input_data[1:], columns=output_fields)
    sheet_df.set_index(['SCENE_ID', 'ZONE_NAME'], inplace=True, drop=True)
    sheet_df['DATE'] = pd.to_datetime(sheet_df['DATE'], format='%m/%d/%Y')

    # Build zone list
    zones = []
    for z_ftr in zones_geojson['features']:
        zone_name = str(
            z_ftr['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_')
        zones.append({
            'name': zone_name,
            'fid': int(z_ftr['id']),
            'csv': os.path.join(
                ini['ZONAL_STATS']['output_ws'], zone_name,
                '{}_landsat_daily.csv'.format(zone_name))})

    # Read in all output CSV files
    logging.debug('\n  Reading CSV files')
    zone_df_list = []
    for zone in zones:
        logging.debug('  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))

        # Build output folder if necessary
        if not os.path.isdir(os.path.dirname(zone['csv'])):
            os.makedirs(os.path.dirname(zone['csv']))

        # Make copy of export field list in order to retain existing columns
        # DEADBEEF - This won't work correctly in the zone loop
        output_fields = export_fields[:]

        # Read existing output table if possible
        logging.debug('    Reading CSV')
        logging.debug('    {}'.format(zone['csv']))
        try:
            zone_df = pd.read_csv(zone['csv'], parse_dates=['DATE'])
            zone_df['DATE'] = zone_df['DATE'].dt.strftime('%Y-%m-%d')
            # zone_df['DATE'] = zone_df['DATE'].dt.strftime('%Y-%m-%d')
            # Move any existing columns not in export_fields to end of CSV
            # output_fields.extend([
            #     f for f in zone_df.columns.values if f not in export_fields])
            # zone_df = zone_df.reindex(columns=output_fields)
            # zone_df.sort_values(by=['DATE', 'ROW'], inplace=True)
            zone_df_list.append(zone_df)
        except IOError:
            logging.debug('    Output path doesn\'t exist, skipping')
        except AttributeError:
            logging.debug('    Output CSV appears to be empty')
        except Exception as e:
            logging.exception(
                '    ERROR: Unhandled Exception\n    {}'.format(e))
            input('ENTER')

    # Combine separate zone dataframes
    try:
        stats_df = pd.concat(zone_df_list)
    except ValueError:
        logging.debug(
            '    Output path(s) doesn\'t exist, building empty dataframe')
        stats_df = pd.DataFrame(columns=export_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')
    del zone_df_list

    # stats_df.set_index(['ZONE_NAME', 'SCENE_ID'], inplace=True, drop=True)
    stats_df.set_index(['SCENE_ID', 'ZONE_NAME'], inplace=True, drop=True)

    if overwrite_flag:
        logging.error('\nOverwrite is not currently supported')
        sys.exit()
        # Delete the matching rows
    else:
        # Append any missing values directly to the Google Sheet
        update_df = stats_df[~stats_df.index.isin(sheet_df.index)]
        logging.debug(update_df.head())

        logging.debug('  Saving data')
        gsheet_writer(update_df, output_fields, ini['GSHEET']['gsheet_id'],
                      ini['GSHEET']['landsat_daily'])




    # # Initialize the Landsat object
    # # For getting SCENE_ID lists, don't use zone_geom or products
    # #   and set mosaic_method to 'none' to get separate SCENE_ID lists
    # #   for each tile
    # # These will be applied below
    # landsat_args = {
    #     k: v for section in ['INPUTS']
    #     for k, v in ini[section].items()
    #     if k in [
    #         'landsat4_flag', 'landsat5_flag',
    #         'landsat7_flag', 'landsat8_flag',
    #         'fmask_flag', 'acca_flag',
    #         'start_year', 'end_year',
    #         'start_month', 'end_month',
    #         'start_doy', 'end_doy',
    #         'scene_id_keep_list', 'scene_id_skip_list',
    #         'path_keep_list', 'row_keep_list',
    #         'refl_sur_method', 'adjust_method', 'mosaic_method']}
    # landsat = ee_common.Landsat(landsat_args)
    # if ini['INPUTS']['tile_geom']:
    #     landsat.tile_geom = ini['INPUTS']['tile_geom']
    # landsat.zone_geom = None
    # landsat.products = []
    # landsat.mosaic_method = 'none'
    #
    # # Build a master list of available SCENE_IDs for each path/row
    # #   based on the INI parameters (set above)
    # # The tile_scene_json dictionary starts off empty but is populated while
    # #   iterating through the tiles.
    # output_ids = set()
    # logging.debug('\n  Building tiles scene lists')
    # for tile in output_tiles:
    #     logging.debug('  {}'.format(tile))
    #     if tile in ini['ZONAL_STATS']['tile_scene_json']:
    #         # Read the previously computed tile SCENE_ID list
    #         output_ids.update(
    #             ini['ZONAL_STATS']['tile_scene_json'][tile])
    #     else:
    #         # Compute the SCENE_ID list for each tile if needed
    #         path_row_re = re.compile('p(?P<PATH>\d{1,3})r(?P<ROW>\d{1,3})')
    #         path, row = list(map(int, path_row_re.match(tile).groups()))
    #
    #         # Filter the Landsat collection down to a single tile
    #         landsat.zone_geom = None
    #         landsat.products = []
    #         landsat.mosaic_method = 'none'
    #         landsat.path_keep_list = [path]
    #         landsat.row_keep_list = [row]
    #         landsat_coll = landsat.get_collection()
    #
    #         # Get new scene ID list
    #         ini['ZONAL_STATS']['tile_scene_json'][tile] = list(utils.ee_getinfo(
    #             landsat_coll.aggregate_histogram('SCENE_ID')).keys())
    #         output_ids.update(ini['ZONAL_STATS']['tile_scene_json'][tile])
    #     logging.debug('  {}'.format(
    #         ', '.join(ini['ZONAL_STATS']['tile_scene_json'][tile])))
    #
    # # If export_ids is empty, all SCENE_IDs may have been filtered
    # if not output_ids:
    #     logging.info('\n  No SCENE_IDs to process after applying INI filters')
    #     return False
    #
    # # Compute mosaiced SCENE_IDs after filtering
    # if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
    #     mosaic_id_dict = defaultdict(list)
    #     for scene_id in output_ids:
    #         mosaic_id = '{}XXX{}'.format(scene_id[:8], scene_id[11:])
    #         mosaic_id_dict[mosaic_id].append(scene_id)
    #     export_ids = set(mosaic_id_dict.keys())
    #
    # # For overwrite, drop all expected entries from existing output DF
    # if overwrite_flag:
    #     output_df = output_df[
    #         ~output_df.index.get_level_values('SCENE_ID').isin(list(export_ids))]
    #     # DEADBEEF - This doesn't work for some rearson
    #     # output_df.drop(
    #     #     output_df.index.get_level_values('SCENE_ID').isin(list(export_ids)),
    #     #     inplace=True)
    #
    # # zone_df below is intentionally being made as a subset copy of output_df
    # # This raises a warning though
    # pd.options.mode.chained_assignment = None  # default='warn'
    #
    # # Add empty entries separately for each zone
    # logging.debug('\n  Identifying scenes/zones with missing data')
    # for zone in zones:
    #     logging.debug('  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))
    #
    #     # Compute custom SCENE_ID lists for each zone
    #     # This could probably simplified or moved out of the loop
    #     export_ids = set()
    #     for zone_tile in ini['ZONAL_STATS']['zone_tile_json'][zone['name']]:
    #         if zone_tile in ini['ZONAL_STATS']['tile_scene_json'].keys():
    #             for id in ini['ZONAL_STATS']['tile_scene_json'][zone_tile]:
    #                 if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
    #                     export_ids.add('{}XXX{}'.format(id[:8], id[11:]))
    #                 else:
    #                     export_ids.add(id)
    #     if not export_ids:
    #         logging.info('    No SCENE_IDs to process, skipping zone')
    #         continue
    #
    #     # Subset the data frame
    #     zone_df = output_df.iloc[
    #         output_df.index.get_level_values('ZONE_NAME')==zone['name']]
    #     zone_df.reset_index(inplace=True)
    #     zone_df.set_index(['SCENE_ID'], inplace=True, drop=True)
    #
    #     # Get list of existing dates in the CSV
    #     if not zone_df.empty:
    #         zone_df_ids = set(zone_df.index.values)
    #     else:
    #         zone_df_ids = set()
    #     logging.debug('    Scenes in zone CSV: {}'.format(
    #         ', '.join(zone_df_ids)))
    #
    #     # List of SCENE_IDs that are entirely missing
    #     # This may include scenes that don't intersect the zone
    #     missing_all_ids = export_ids - zone_df_ids
    #     logging.debug('    Scenes not in CSV (missing all values): {}'.format(
    #         ', '.join(sorted(missing_all_ids))))
    #     if not missing_all_ids:
    #         continue
    #
    #     # Get a list of SCENE_IDs that intersect the zone
    #     # Get SCENE_ID list mimicking a full extract below
    #     #   (but without products)
    #     # Start with INI path/row keep list but update based on SCENE_ID later
    #     landsat.products = []
    #     landsat.path_keep_list = landsat_args['path_keep_list']
    #     landsat.row_keep_list = landsat_args['row_keep_list']
    #     landsat.zone_geom = zone['ee_geom']
    #     landsat.mosaic_method = landsat_args['mosaic_method']
    #
    #     # Only use the scene_id_keep_list if there was already data in the DF
    #     if zone_df.index.values.any():
    #         # SCENE_ID keep list must be non-mosaiced IDs
    #         if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
    #             landsat.scene_id_keep_list = sorted(set([
    #                 scene_id for mosaic_id in missing_all_ids
    #                 for scene_id in mosaic_id_dict[mosaic_id]]))
    #         else:
    #             landsat.scene_id_keep_list = sorted(missing_all_ids)
    #         landsat.set_landsat_from_scene_id()
    #         landsat.set_tiles_from_scene_id()
    #
    #     # Get the SCENE_IDs that intersect the zone
    #     # Process each Landsat type independently
    #     # Was having issues getting the full scene list at once
    #     logging.debug('    Getting intersecting SCENE_IDs')
    #     missing_zone_ids = set()
    #     landsat_type_list = landsat._landsat_list[:]
    #     for landsat_str in landsat_type_list:
    #         landsat._landsat_list = [landsat_str]
    #         missing_zone_ids.update(set(utils.ee_getinfo(
    #             landsat.get_collection().aggregate_histogram('SCENE_ID'))))
    #         # logging.debug('      {} {}'.format(
    #         #     landsat_str, len(missing_zone_ids)))
    #     landsat._landsat_list = landsat_type_list
    #
    #     # # Get the SCENE_IDs that intersect the zone
    #     # logging.debug('    Getting intersecting SCENE_IDs')
    #     missing_zone_ids = set(utils.ee_getinfo(
    #         landsat.get_collection().aggregate_histogram('SCENE_ID')))
    #
    #     # Difference of sets are SCENE_IDs that don't intersect
    #     missing_nonzone_ids = missing_all_ids - missing_zone_ids
    #
    #     # Remove skipped/empty SCENE_IDs from possible SCENE_ID list
    #     # DEADBEEF - This is modifying the master export_id list
    #     #   Commenting out for now
    #     # export_ids = export_ids - missing_skip_ids
    #     logging.debug('    Scenes intersecting zone: {}'.format(
    #         ', '.join(sorted(missing_zone_ids))))
    #     logging.debug('    Scenes not intersecting zone: {}'.format(
    #         ', '.join(sorted(missing_nonzone_ids))))
    #     logging.info('    Include ID count: {}'.format(
    #         len(missing_zone_ids)))
    #     logging.info('    Exclude ID count: {}'.format(
    #         len(missing_nonzone_ids)))
    #
    #     # Create empty entries for all scenes that are missing
    #     # Identify whether the missing scenes intersect the zone or not
    #     # Set pixels counts for non-intersecting SCENE_IDs to 0
    #     # Set pixel counts for intersecting SCENE_IDs to NaN
    #     #   This will allow the zonal stats to set actual pixel count value
    #     logging.debug('    Appending all empty SCENE_IDs')
    #     missing_df = pd.DataFrame(
    #         index=missing_all_ids, columns=output_df.columns)
    #     missing_df.index.name = 'SCENE_ID'
    #     missing_df['ZONE_NAME'] = zone['name']
    #     missing_df['ZONE_FID'] = zone['fid']
    #     # missing_df['AREA'] = zone['area']
    #     missing_df['PLATFORM'] = missing_df.index.str.slice(0, 4)
    #     missing_df['PATH'] = missing_df.index.str.slice(5, 8).astype(int)
    #     missing_df['DATE'] = pd.to_datetime(
    #         missing_df.index.str.slice(12, 20), format='%Y%m%d')
    #     missing_df['YEAR'] = missing_df['DATE'].dt.year
    #     missing_df['MONTH'] = missing_df['DATE'].dt.month
    #     missing_df['DAY'] = missing_df['DATE'].dt.day
    #     missing_df['DOY'] = missing_df['DATE'].dt.dayofyear.astype(int)
    #     missing_df['DATE'] = missing_df['DATE'].dt.strftime('%Y-%m-%d')
    #     # DEADBEEF - Setting ROW to np.nan will cause problems in QAQ and
    #     #   summary tables unless fixed later on
    #     # ROW needs to be set to NAN so that update call will replace it
    #     missing_df['ROW'] = np.nan
    #     missing_df['QA'] = np.nan
    #     missing_df['PIXEL_SIZE'] = landsat.cellsize
    #     missing_df['PIXEL_COUNT'] = np.nan
    #     missing_df['PIXEL_TOTAL'] = np.nan
    #     missing_df['FMASK_COUNT'] = np.nan
    #     missing_df['FMASK_TOTAL'] = np.nan
    #     missing_df['FMASK_PCT'] = np.nan
    #     if 'etstar_mean' in landsat.products:
    #         missing_df['ETSTAR_COUNT'] = np.nan
    #     missing_df['CLOUD_SCORE'] = np.nan
    #
    #     # Set intersecting zone pixel counts to 0
    #     if missing_nonzone_ids:
    #         missing_nonzone_mask = missing_df.index.isin(missing_nonzone_ids)
    #         missing_df.loc[missing_nonzone_mask, 'ROW'] = 0
    #         missing_df.loc[missing_nonzone_mask, 'PIXEL_COUNT'] = 0
    #         missing_df.loc[missing_nonzone_mask, 'PIXEL_TOTAL'] = 0
    #         missing_df.loc[missing_nonzone_mask, 'FMASK_COUNT'] = 0
    #         missing_df.loc[missing_nonzone_mask, 'FMASK_TOTAL'] = 0
    #
    #     # Remove the overlapping missing entries
    #     # Then append the new missing entries to the zone CSV
    #     # if zone_df.index.intersection(missing_df.index).any():
    #     try:
    #         zone_df.drop(
    #             zone_df.index.intersection(missing_df.index),
    #             inplace=True)
    #     except ValueError:
    #         pass
    #     zone_df = zone_df.append(missing_df)
    #     logging.debug('    Writing to CSV')
    #     csv_writer(zone_df, zone['csv'], export_fields)
    #
    #     # Update the master dataframe
    #     zone_df.reset_index(inplace=True)
    #     zone_df.set_index(
    #         ['ZONE_NAME', 'SCENE_ID'], inplace=True, drop=True)
    #     try:
    #         output_df.drop(
    #             output_df.index.get_level_values('ZONE_NAME')==zone['name'],
    #             inplace=True)
    #     except Exception as e:
    #         # These seem to happen with the zone is not in the output_df
    #         logging.debug('    Exception: {}'.format(e))
    #         pass
    #     output_df = output_df.append(zone_df)
    #
    # # Putting the warning back to the default balue
    # pd.options.mode.chained_assignment = 'warn'
    #
    # # Identify SCENE_IDs that are missing any data
    # # Filter based on product and SCENE_ID lists
    # missing_id_mask = output_df.index.get_level_values('SCENE_ID') \
    #     .isin(export_ids)
    # # missing_id_mask = (
    # #     (output_df['PIXEL_COUNT'] != 0) &
    # #     output_df.index.get_level_values('SCENE_ID').isin(export_ids))
    # products = [f.upper() for f in ini['ZONAL_STATS']['landsat_products']]
    # missing_df = output_df.loc[missing_id_mask, products].isnull()
    #
    # # List of SCENE_IDs and products with some missing data
    # missing_ids = set(missing_df[missing_df.any(axis=1)]
    #                       .index.get_level_values('SCENE_ID').values)
    #
    # # Skip processing if all dates already exist in the CSV
    # if not missing_ids and not overwrite_flag:
    #     logging.info('\n  All scenes present, returning')
    #     return True
    # else:
    #     logging.debug('\n  Scenes missing values: {}'.format(
    #         ', '.join(sorted(missing_ids))))
    #
    # # Reset the Landsat collection args
    # landsat.path_keep_list = landsat_args['path_keep_list']
    # landsat.row_keep_list = landsat_args['row_keep_list']
    # landsat.products = ini['ZONAL_STATS']['landsat_products']
    # landsat.mosaic_method = landsat_args['mosaic_method']
    #
    # def export_update(data_df):
    #     """Set/modify ancillary field values in the export CSV dataframe"""
    #     # First, remove any extra rows that were added for exporting
    #     # The DEADBEEF entry is added because the export structure is based on
    #     #   the first feature in the collection, so fields with nodata will
    #     #   be excluded
    #     data_df.drop(
    #         data_df[data_df['SCENE_ID'] == 'DEADBEEF'].index,
    #         inplace=True)
    #
    #     # Add additional fields to the export data frame
    #     if not data_df.empty:
    #         data_df['PLATFORM'] = data_df['SCENE_ID'].str.slice(0, 4)
    #         data_df['PATH'] = data_df['SCENE_ID'].str.slice(5, 8).astype(int)
    #         data_df['DATE'] = pd.to_datetime(
    #             data_df['SCENE_ID'].str.slice(12, 20), format='%Y%m%d')
    #         data_df['YEAR'] = data_df['DATE'].dt.year
    #         data_df['MONTH'] = data_df['DATE'].dt.month
    #         data_df['DAY'] = data_df['DATE'].dt.day
    #         data_df['DOY'] = data_df['DATE'].dt.dayofyear.astype(int)
    #         data_df['DATE'] = data_df['DATE'].dt.strftime('%Y-%m-%d')
    #         data_df['AREA'] = data_df['PIXEL_COUNT'] * landsat.cellsize ** 2
    #         data_df['PIXEL_SIZE'] = landsat.cellsize
    #
    #         fmask_mask = data_df['FMASK_TOTAL'] > 0
    #         if fmask_mask.any():
    #             data_df.loc[fmask_mask, 'FMASK_PCT'] = 100.0 * (
    #                 data_df.loc[fmask_mask, 'FMASK_COUNT'] /
    #                 data_df.loc[fmask_mask, 'FMASK_TOTAL'])
    #         data_df['QA'] = 0
    #
    #     # Remove unused export fields
    #     if 'system:index' in data_df.columns.values:
    #         del data_df['system:index']
    #     if '.geo' in data_df.columns.values:
    #         del data_df['.geo']
    #
    #     data_df.set_index(
    #         ['ZONE_NAME', 'SCENE_ID'], inplace=True, drop=True)
    #     return data_df
    #
    #     # Save updated CSV
    #     # if output_df is not None and not output_df.empty:
    #     if not output_df.empty:
    #         logging.info('\n  Writing zone CSVs')
    #         for zone in zones:
    #             logging.debug(
    #                 '  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))
    #             # logging.debug('    {}'.format(zone_output_path))
    #             zone_df = output_df.iloc[
    #                 output_df.index.get_level_values('ZONE_NAME')==zone['name']]
    #             csv_writer(zone_df, zone['csv'], export_fields)
    #     else:
    #         logging.info('\n  Empty output dataframe')


def gridmet_daily_func(gsheet_cred, export_fields, ini, zones_geojson,
                       overwrite_flag=False):
    """

    Parameters
    ----------
    gsheet_cred
    export_fields : list
    ini : dict
        Input file parameters.
    zones_geojson : dict
        Zone specific parameters.
    overwrite_flag : bool
        If True, overwrite existing files.

    """

    logging.info('\nGRIDMET Daily ETo/PPT')

    gridmet_products = ini['ZONAL_STATS']['gridmet_products'][:]
    gridmet_fields = [f.upper() for f in gridmet_products]

    # Assuming Google Sheet exists and has the target columns
    # Assuming worksheet is called "Landsat_Daily"
    logging.info('    Reading Landsat GSHEET')
    gsheet = gsheet_cred.open_by_key(ini['GSHEET']['gsheet_id'])\
        .worksheet(ini['GSHEET']['gridmet_daily'])
    output_fields = gsheet.row_values(1)

    # Read in full data frame
    input_df = pd.DataFrame(
        data=gsheet.get_all_values()[1:], columns=output_fields)
    input_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)

    # List of all zone names to iterate over
    zone_list = []
    for z_ftr in zones_geojson['features']:
        zone_name = str(
            z_ftr['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_')
        zone_output_path = os.path.join(
            ini['ZONAL_STATS']['output_ws'], zone_name,
            '{}_gridmet_daily.csv'.format(zone_name))
        zone_list.append([zone_name, int(z_ftr['id']), zone_output_path])

    # Read in existing data if possible
    logging.debug('\n  Reading existing CSV files')
    zone_df_list = []
    for zone_name, zone_fid, zone_output_path in zone_list:
        logging.debug('  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
        logging.debug('    {}'.format(zone_output_path))
        if not os.path.isdir(os.path.dirname(zone_output_path)):
            os.makedirs(os.path.dirname(zone_output_path))

        # Read in existing data if possible
        try:
            zone_df = pd.read_csv(zone_output_path, parse_dates=['DATE'])
            zone_df['DATE'] = zone_df['DATE'].dt.strftime('%Y-%m-%d')
            zone_df_list.append(zone_df)
        except IOError:
            logging.debug('    Output path doesn\'t exist, skipping')
        except Exception as e:
            logging.exception(
                '    ERROR: Unhandled Exception\n    {}'.format(e))
            input('ENTER')

    # Combine separate zone dataframes
    try:
        output_df = pd.concat(zone_df_list)
    except ValueError:
        logging.debug(
            '    Output path(s) doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=export_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')
    del zone_df_list

    # Use the date string as the index
    output_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)






    # # Get list of possible dates based on INI
    # export_dates = set(
    #     date_str for date_str in utils.date_range(
    #         '{}-01-01'.format(ini['INPUTS']['start_year'] - 1),
    #         '{}-12-31'.format(ini['INPUTS']['end_year']))
    #     if datetime.datetime.strptime(date_str, '%Y-%m-%d') <= gridmet_end_dt)
    # # logging.debug('  Export Dates: {}'.format(
    # #     ', '.join(sorted(export_dates))))
    #
    # # For overwrite, drop all expected entries from existing output DF
    # if overwrite_flag:
    #     output_df = output_df[
    #         ~output_df.index.get_level_values('DATE').isin(list(export_dates))]
    #
    # # Identify missing dates in any zone
    # # Iterate by zone in order to set zone name/fid
    # logging.debug('\n  Identifying dates/zones with missing data')
    # missing_dates = set()
    # for zone_name, zone_fid, zone_output_path in zone_list:
    #     logging.debug('  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
    #     zone_df = output_df.iloc[
    #         output_df.index.get_level_values('ZONE_NAME') == zone_name]
    #     zone_df.reset_index(inplace=True)
    #     zone_df.set_index(['DATE'], inplace=True, drop=True)
    #
    #     # Get list of existing dates in the CSV
    #     if not zone_df.empty:
    #         zone_dates = set(zone_df.index.values)
    #     else:
    #         zone_dates = set()
    #
    #     missing_all_dates = export_dates - zone_dates
    #     missing_dates.update(missing_all_dates)
    #     # logging.debug('    Dates missing all values: {}'.format(
    #     #     ', '.join(sorted(missing_all_dates))))
    #
    #     # Add non-intersecting SCENE_IDs directly to the output dataframe
    #     if missing_all_dates:
    #         logging.debug('    Appending missing dates')
    #         missing_df = pd.DataFrame(
    #             index=missing_all_dates, columns=output_df.columns)
    #         missing_index = pd.to_datetime(missing_df.index, format='%Y-%m-%d')
    #         missing_df.index.name = 'DATE'
    #         missing_df['ZONE_NAME'] = zone_name
    #         missing_df['ZONE_FID'] = zone_fid
    #         missing_df['YEAR'] = missing_index.year
    #         missing_df['MONTH'] = missing_index.month
    #         missing_df['DAY'] = missing_index.day
    #         missing_df['DOY'] = missing_index.dayofyear.astype(int)
    #         # Build the datetime for the start of the month
    #         # Move the datetime forward 3 months
    #         # Get the year
    #         missing_df['WATER_YEAR'] = (
    #             pd.to_datetime(
    #                 missing_index.strftime('%Y-%m-01'), format='%Y-%m-%d') +
    #             pd.DateOffset(months=3)).year
    #
    #         # Remove the overlapping missing entries
    #         # Then append the new missing entries to the zone CSV
    #         # if zone_df.index.intersection(missing_df.index).any():
    #         try:
    #             zone_df.drop(
    #                 zone_df.index.intersection(missing_df.index), inplace=True)
    #         except ValueError:
    #             pass
    #         zone_df = zone_df.append(missing_df)
    #         csv_writer(zone_df, zone_output_path, export_fields)
    #
    #         # Update the master dataframe
    #         zone_df.reset_index(inplace=True)
    #         zone_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)
    #         try:
    #             output_df.drop(
    #                 output_df.index.get_level_values('ZONE_NAME') == zone_name,
    #                 inplace=True)
    #         except (ValueError, KeyError):
    #             # These seem to happen with the zone is not in the output_df
    #             pass
    #         output_df = output_df.append(zone_df)
    #
    # # Identify SCENE_IDs that are missing any data
    # # Filter based on product and SCENE_ID lists
    # missing_date_mask = output_df.index.get_level_values('DATE') \
    #     .isin(export_dates)
    # missing_df = output_df.loc[missing_date_mask, gridmet_fields].isnull()
    #
    # # List of SCENE_IDs and products with some missing data
    # missing_any_dates = set(missing_df[
    #     missing_df.any(axis=1)].index.get_level_values('DATE').values)
    # # logging.debug('\n  Dates missing any values: {}'.format(
    # #     ', '.join(sorted(missing_any_dates))))
    #
    # missing_dates.update(missing_any_dates)
    # # logging.debug('\n  Dates missing values: {}'.format(
    # #     ', '.join(sorted(missing_dates))))
    #
    # # Skip processing if all dates already exist in the CSV
    # if not missing_dates and not overwrite_flag:
    #     logging.info('\n  All dates present, returning')
    #     return True
    # export_date_list = sorted(missing_dates)
    #
    # # Update/limit GRIDMET products list if necessary
    # if not missing_df.empty:
    #     gridmet_products = set(
    #         f.lower()
    #         for f in missing_df.columns[missing_df.any(axis=0)]
    #         if f.lower() in gridmet_products)
    #     logging.debug('\n  Products missing any values: {}'.format(
    #         ', '.join(sorted(gridmet_products))))
    #
    # # Identify zones with missing data
    # missing_zones = set(
    #     missing_df[missing_df.any(axis=1)].index.get_level_values('ZONE_NAME'))
    # logging.debug('\n  Zones with missing data: {}'.format(
    #     ', '.join(sorted(missing_zones))))
    #
    # # Build collection of all features to test for each SCENE_ID
    # # I have to build a geometry in order to set a non-WGS84 projection
    # # Limit zones collection to only zones with missing data
    # zone_ftr_list = []
    # for z in zones_geojson['features']:
    #     zone_name = str(z['properties'][ini['INPUTS']['zone_field']]) \
    #         .replace(' ', '_')
    #     if zone_name not in missing_zones:
    #         continue
    #     zone_ftr_list.append(ee.Feature(
    #         ee.Geometry(
    #             geo_json=z['geometry'], opt_proj=zones_wkt, opt_geodesic=False),
    #         {
    #             'ZONE_NAME': zone_name,
    #             'ZONE_FID': int(z['id'])
    #         }))
    # zone_coll = ee.FeatureCollection(zone_ftr_list)
    #
    #     # Save updated CSVs
    #     if not output_df.empty:
    #         logging.info('\n  Writing zone CSVs')
    #         for zone_name, zone_fid, zone_output_path in zone_list:
    #             logging.debug(
    #                 '  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
    #             # logging.debug('    {}'.format(zone_output_path))
    #             if zone_name not in missing_zones:
    #                 logging.debug('    No missing values, skipping zone')
    #                 continue
    #
    #             zone_df = output_df.iloc[
    #                 output_df.index.get_level_values('ZONE_NAME') == zone_name]
    #             zone_df.reset_index(inplace=True)
    #             zone_df.set_index(['DATE'], inplace=True, drop=True)
    #             if zone_df.empty:
    #                 logging.debug('    Empty zone df, skipping')
    #                 continue
    #             csv_writer(zone_df, zone_output_path, export_fields)


def gridmet_monthly_func(gsheet_cred, export_fields, ini, zones_geojson,
                         overwrite_flag=False):
    """

    Parameters
    ----------
    gsheet_cred
    export_fields : list
    ini : dict
        Input file parameters.
    zones_geojson : dict
        Zone specific parameters.
    overwrite_flag : bool
        If True, overwrite existing files.
    """

    logging.info('\nGRIDMET Monthly ETo/PPT')

    gridmet_products = ini['ZONAL_STATS']['gridmet_products'][:]
    gridmet_fields = [f.upper() for f in gridmet_products]

    # Assuming Google Sheet exists and has the target columns
    # Assuming worksheet is called "Landsat_Daily"
    logging.info('    Reading Landsat GSHEET')
    gsheet = gsheet_cred.open_by_key(ini['GSHEET']['gsheet_id'])\
        .worksheet(ini['GSHEET']['gridmet_daily'])
    output_fields = gsheet.row_values(1)

    # Read in full data frame
    input_df = pd.DataFrame(
        data=gsheet.get_all_values()[1:], columns=output_fields)
    input_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)

    # List of all zone names to iterate over
    zone_list = []
    for z_ftr in zones_geojson['features']:
        zone_name = str(
            z_ftr['properties'][ini['INPUTS']['zone_field']]).replace(' ', '_')
        zone_output_path = os.path.join(
            ini['ZONAL_STATS']['output_ws'], zone_name,
            '{}_gridmet_monthly.csv'.format(zone_name))
        zone_list.append([zone_name, int(z_ftr['id']), zone_output_path])

    # Read in existing data if possible
    logging.debug('\n  Reading existing CSV files')
    zone_df_list = []
    for zone_name, zone_fid, zone_output_path in zone_list:
        logging.debug('  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
        logging.debug('    {}'.format(zone_output_path))
        if not os.path.isdir(os.path.dirname(zone_output_path)):
            os.makedirs(os.path.dirname(zone_output_path))

        # Read in existing data if possible
        try:
            zone_df = pd.read_csv(zone_output_path, parse_dates=['DATE'])
            zone_df['DATE'] = zone_df['DATE'].dt.strftime('%Y-%m-%d')
            zone_df_list.append(zone_df)
        except IOError:
            logging.debug('    Output path doesn\'t exist, skipping')
        except Exception as e:
            logging.exception(
                '    ERROR: Unhandled Exception\n    {}'.format(e))
            input('ENTER')

    # Combine separate zone dataframes
    try:
        output_df = pd.concat(zone_df_list)
    except ValueError:
        logging.debug(
            '    Output path(s) doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=export_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')
    del zone_df_list

    # Use the date string as the index
    output_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)








    # # Get list of possible dates based on INI
    # export_dates = set([
    #     datetime.datetime(y, m, 1).strftime('%Y-%m-%d')
    #     for y in range(
    #         ini['INPUTS']['start_year'] - 1, ini['INPUTS']['end_year'] + 1)
    # for m in range(1, 13)
    #     if datetime.datetime(y, m, 1) <= gridmet_end_dt])
    # # logging.debug('  Export Dates: {}'.format(
    # #     ', '.join(sorted(export_dates))))
    #
    # # For overwrite, drop all expected entries from existing output DF
    # if overwrite_flag:
    #     output_df = output_df[
    #         ~output_df.index.get_level_values('DATE').isin(list(export_dates))]
    #
    # # Identify missing dates in any zone
    # # Iterate by zone in order to set zone name/fid
    # logging.debug('\n  Identifying dates/zones with missing data')
    # missing_dates = set()
    # for zone_name, zone_fid, zone_output_path in zone_list:
    #     logging.debug('  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
    #     zone_df = output_df.iloc[
    #         output_df.index.get_level_values('ZONE_NAME') == zone_name]
    #     zone_df.reset_index(inplace=True)
    #     zone_df.set_index(['DATE'], inplace=True, drop=True)
    #
    #     # Get list of existing dates in the CSV
    #     if not zone_df.empty:
    #         zone_dates = set(zone_df.index.values)
    #     else:
    #         zone_dates = set()
    #
    #     missing_all_dates = export_dates - zone_dates
    #     missing_dates.update(missing_all_dates)
    #     # logging.debug('    Dates missing all values: {}'.format(
    #     #     ', '.join(sorted(missing_all_dates))))
    #
    #     # Add non-intersecting SCENE_IDs directly to the output dataframe
    #     if missing_all_dates:
    #         logging.debug('    Appending missing dates')
    #         missing_df = pd.DataFrame(
    #             index=missing_all_dates, columns=output_df.columns)
    #         missing_index = pd.to_datetime(missing_df.index, format='%Y-%m-%d')
    #         missing_df.index.name = 'DATE'
    #         missing_df['ZONE_NAME'] = zone_name
    #         missing_df['ZONE_FID'] = zone_fid
    #         missing_df['YEAR'] = missing_index.year
    #         missing_df['MONTH'] = missing_index.month
    #         # missing_df['DAY'] = missing_index.day
    #         # missing_df['DOY'] = missing_index.dayofyear.astype(int)
    #         # Build the datetime for the start of the month
    #         # Move the datetime forward 3 months
    #         # Get the year
    #         missing_df['WATER_YEAR'] = (
    #             pd.to_datetime(
    #                 missing_index.strftime('%Y-%m-01'), format='%Y-%m-%d') +
    #             pd.DateOffset(months=3)).year
    #
    #         # Remove the overlapping missing entries
    #         # Then append the new missing entries to the zone CSV
    #         # if zone_df.index.intersection(missing_df.index).any():
    #         try:
    #             zone_df.drop(
    #                 zone_df.index.intersection(missing_df.index), inplace=True)
    #         except ValueError:
    #             pass
    #         zone_df = zone_df.append(missing_df)
    #         csv_writer(zone_df, zone_output_path, export_fields)
    #
    #         # Update the master dataframe
    #         zone_df.reset_index(inplace=True)
    #         zone_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)
    #         try:
    #             output_df.drop(
    #                 output_df.index.get_level_values('ZONE_NAME') == zone_name,
    #                 inplace=True)
    #         except (ValueError, KeyError):
    #             # These seem to happen with the zone is not in the output_df
    #             pass
    #         output_df = output_df.append(zone_df)
    #
    # # Identify SCENE_IDs that are missing any data
    # # Filter based on product and SCENE_ID lists
    # missing_date_mask = output_df.index.get_level_values('DATE')\
    #     .isin(export_dates)
    # missing_df = output_df.loc[missing_date_mask, gridmet_fields].isnull()
    #
    # # List of SCENE_IDs and products with some missing data
    # missing_any_dates = set(missing_df[
    #     missing_df.any(axis=1)].index.get_level_values('DATE').values)
    # # logging.debug('\n  Dates missing any values: {}'.format(
    # #     ', '.join(sorted(missing_any_dates))))
    #
    # missing_dates.update(missing_any_dates)
    # logging.debug('\n  Dates missing values: {}'.format(
    #     ', '.join(sorted(missing_dates))))
    #
    # # Skip processing if all dates already exist in the CSV
    # if not missing_dates and not overwrite_flag:
    #     logging.info('\n  All dates present, returning')
    #     return True
    # export_date_list = sorted(missing_dates)
    #
    # # Update/limit GRIDMET products list if necessary
    # if not missing_df.empty:
    #     gridmet_products = set(
    #         f.lower()
    #         for f in missing_df.columns[missing_df.any(axis=0)]
    #         if f.lower() in gridmet_products)
    #     logging.debug('\n  Products missing any values: {}'.format(
    #         ', '.join(sorted(gridmet_products))))
    #
    # # Identify zones with missing data
    # missing_zones = set(
    #     missing_df[missing_df.any(axis=1)].index.get_level_values('ZONE_NAME'))
    # logging.debug('\n  Zones with missing data: {}'.format(
    #     ', '.join(sorted(missing_zones))))
    #
    # # Build collection of all features to test for each SCENE_ID
    # # I have to build a geometry in order to set a non-WGS84 projection
    # # Limit zones collection to only zones with missing data
    # zone_ftr_list = []
    # for z in zones_geojson['features']:
    #     zone_name = str(z['properties'][ini['INPUTS']['zone_field']])\
    #         .replace(' ', '_')
    #     if zone_name not in missing_zones:
    #         continue
    #     zone_ftr_list.append(ee.Feature(
    #         ee.Geometry(
    #             geo_json=z['geometry'], opt_proj=zones_wkt, opt_geodesic=False),
    #         {
    #             'ZONE_NAME': zone_name,
    #             'ZONE_FID': int(z['id'])
    #         }))
    # zone_coll = ee.FeatureCollection(zone_ftr_list)
    #
    # # Save updated CSVs
    # if not output_df.empty:
    #     logging.info('\n  Writing zone CSVs')
    #     for zone_name, zone_fid, zone_output_path in zone_list:
    #         logging.debug('  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
    #         # logging.debug('    {}'.format(zone_output_path))
    #         if zone_name not in missing_zones:
    #             logging.debug('    No missing values, skipping zone')
    #             continue
    #
    #         # zone_df = output_df[output_df['ZONE_NAME']==zone_name]
    #         zone_df = output_df.iloc[
    #             output_df.index.get_level_values('ZONE_NAME') == zone_name]
    #         zone_df.reset_index(inplace=True)
    #         zone_df.set_index(['DATE'], inplace=True, drop=True)
    #         if zone_df.empty:
    #             logging.debug('    Empty zone df, skipping')
    #             continue
    #         csv_writer(zone_df, zone_output_path, export_fields)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Export zonal stats to Google Sheet',
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
