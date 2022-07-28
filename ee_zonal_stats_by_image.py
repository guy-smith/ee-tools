#--------------------------------
# Name:         ee_zonal_stats_by_image.py
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
import numpy as np
from osgeo import ogr
import pandas as pd

import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils

pp = pprint.PrettyPrinter(indent=4)


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

    if ini['EXPORT']['export_dest'] != 'getinfo':
        logging.critical('\nERROR: Only GetInfo exports are currently supported\n')
        sys.exit()

    # # Zonal stats init file paths
    # zone_geojson_path = os.path.join(
    #     ini['ZONAL_STATS']['output_ws'],
    #     os.path.basename(ini['INPUTS']['zone_shp_path']).replace(
    #         '.shp', '.geojson'))

    # These may eventually be set in the INI file
    landsat_daily_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'SCENE_ID', 'PLATFORM',
        'PATH', 'ROW', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'AREA', 'PIXEL_SIZE', 'PIXEL_COUNT', 'PIXEL_TOTAL',
        'FMASK_COUNT', 'FMASK_TOTAL', 'FMASK_PCT', 'CLOUD_SCORE', 'QA']
    gridmet_daily_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY',
        'WATER_YEAR', 'ETO', 'PPT']
    gridmet_monthly_fields = [
        'ZONE_NAME', 'ZONE_FID', 'DATE', 'YEAR', 'MONTH', 'WATER_YEAR',
        'ETO', 'PPT']
    # pdsi_dekad_fields = [
    #     'ZONE_NAME', 'ZONE_FID', 'DATE', 'YEAR', 'MONTH', 'DAY', 'DOY',
    #     'PDSI']

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
            landsat_daily_fields, ini, zones_geojson, zones_wkt, overwrite_flag)
    if ini['ZONAL_STATS']['gridmet_daily_flag']:
        gridmet_daily_func(
            gridmet_daily_fields, ini, zones_geojson, zones_wkt, gridmet_end_dt,
            overwrite_flag)
    if ini['ZONAL_STATS']['gridmet_monthly_flag']:
        gridmet_monthly_func(
            gridmet_monthly_fields, ini, zones_geojson, zones_wkt, gridmet_end_dt,
            overwrite_flag)
    # if ini['ZONAL_STATS']['pdsi_flag']:
    #     pdsi_func(pdsi_dekad_fields, ini, zones, zones_wkt,
    #     overwrite_flag)


def landsat_func(export_fields, ini, zones_geojson, zones_wkt,
                 overwrite_flag=False):
    """

    Function will attempt to generate export tasks only for missing SCENE_IDs
    Also try to limit the products to only those with missing data

    Parameters
    ----------
    export_fields : list
    ini : dict
        Input file parameters.
    zones_geojson : dict
        Zones GeoJSON.
    zones_wkt : str
        Zones spatial reference Well Known Text.
    overwrite_flag : bool, optional
        If True, overwrite existing values (the default is False).
        Don't remove/replace the CSV file directly.

    """
    logging.info('\nLandsat')

    # DEADBEEF - For now, hardcode transform to a standard Landsat image
    ini['EXPORT']['transform'] = [30.0, 0.0, 15.0, 0.0, -30.0, 15.0]
    # ini['EXPORT']['transform'] = '[{}]'.format(','.join(
    #     map(str, (30.0, 0.0, 15.0, 0.0, -30.0, 15.0))))
    logging.debug('  Output Transform: {}'.format(
        ini['EXPORT']['transform']))

    landsat_products = ini['ZONAL_STATS']['landsat_products'][:]
    landsat_fields = [f.upper() for f in landsat_products]

    def csv_writer(output_df, output_path, output_fields):
        """Write the dataframe to CSV with custom formatting"""
        csv_df = output_df.copy()

        # Convert float fields to objects, set NaN to None
        float_fields = landsat_fields + ['CLOUD_SCORE', 'FMASK_PCT']
        for field in csv_df.columns.values:
            if field.upper() not in float_fields:
                continue
            csv_df[field] = csv_df[field].astype(object)
            null_mask = csv_df[field].isnull()
            csv_df.loc[null_mask, field] = None
            csv_df.loc[~null_mask, field] = csv_df.loc[~null_mask, field].map(
                lambda x: '{0:10.6f}'.format(x).strip())
            # csv_df.loc[~null_mask, [field]] = csv_df.loc[~null_mask, [field]].apply(
            #     lambda x: '{0:10.6f}'.format(x[0]).strip(), axis=1)

        # Set field types
        # Don't set the following since they may contain NaN/None?
        # 'QA', 'PIXEL_TOTAL', 'PIXEL_COUNT', 'FMASK_TOTAL', 'FMASK_COUNT']
        for field in ['ZONE_FID', 'PATH', 'YEAR', 'MONTH', 'DAY', 'DOY']:
            csv_df[field] = csv_df[field].astype(int)
        # if csv_df['ZONE_NAME'].dtype == np.float64:
        #     csv_df['ZONE_NAME'] = csv_df['ZONE_NAME'].astype(int).astype(str)

        # DEADBEEF
        # if csv_df['QA'].isnull().any():
        #     csv_df.loc[csv_df['QA'].isnull(), 'QA'] = 0
        # fmask_mask = csv_df['FMASK_TOTAL'] > 0
        # if fmask_mask.any():
        #     csv_df.loc[fmask_mask, 'FMASK_PCT'] = 100.0 * (
        #         csv_df.loc[fmask_mask, 'FMASK_COUNT'] /
        #         csv_df.loc[fmask_mask, 'FMASK_TOTAL'])

        csv_df.reset_index(drop=False, inplace=True)
        csv_df.sort_values(by=['DATE', 'ROW'], inplace=True)
        csv_df.to_csv(output_path, index=False, columns=output_fields)

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
    logging.debug('\n  Filtering zone tiles')
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
        output_df = pd.concat(zone_df_list, sort=False)
    except ValueError:
        logging.debug(
            '    Output path(s) doesn\'t exist, building empty dataframe')
        output_df = pd.DataFrame(columns=export_fields)
    except Exception as e:
        logging.exception('    ERROR: Unhandled Exception\n    {}'.format(e))
        input('ENTER')
    del zone_df_list

    output_df.set_index(['ZONE_NAME', 'SCENE_ID'], inplace=True, drop=True)

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

    # For overwrite, drop all expected entries from existing output DF
    if overwrite_flag:
        output_df = output_df[
            ~output_df.index.get_level_values('SCENE_ID').isin(list(export_ids))]
        # DEADBEEF - This doesn't work for some rearson
        # output_df.drop(
        #     output_df.index.get_level_values('SCENE_ID').isin(list(export_ids)),
        #     inplace=True)

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
        zone_df = output_df.iloc[
            output_df.index.get_level_values('ZONE_NAME')==zone['name']]
        zone_df.reset_index(inplace=True)
        zone_df.set_index(['SCENE_ID'], inplace=True, drop=True)

        # Get list of existing dates in the CSV
        if not zone_df.empty:
            zone_df_ids = set(zone_df.index.values)
        else:
            zone_df_ids = set()
        logging.debug('    Scenes in zone CSV: {}'.format(
            ', '.join(zone_df_ids)))

        # List of SCENE_IDs that are entirely missing
        # This may include scenes that don't intersect the zone
        missing_all_ids = export_ids - zone_df_ids
        logging.debug('    Scenes not in CSV (missing all values): {}'.format(
            ', '.join(sorted(missing_all_ids))))
        if not missing_all_ids:
            continue

        # Get a list of SCENE_IDs that intersect the zone
        # Get SCENE_ID list mimicking a full extract below
        #   (but without products)
        # Start with INI path/row keep list but update based on SCENE_ID later
        landsat.products = []
        landsat.path_keep_list = landsat_args['path_keep_list']
        landsat.row_keep_list = landsat_args['row_keep_list']
        landsat.zone_geom = zone['ee_geom']
        landsat.mosaic_method = landsat_args['mosaic_method']

        # Only use the scene_id_keep_list if there was already data in the DF
        if zone_df.index.values.any():
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
            # logging.debug('      {} {}'.format(
            #     landsat_str, len(missing_zone_ids)))
        landsat._landsat_list = landsat_type_list

        # # Get the SCENE_IDs that intersect the zone
        # logging.debug('    Getting intersecting SCENE_IDs')
        missing_zone_ids = set(utils.ee_getinfo(
            landsat.get_collection().aggregate_histogram('SCENE_ID')))

        # Difference of sets are SCENE_IDs that don't intersect
        missing_nonzone_ids = missing_all_ids - missing_zone_ids

        # Remove skipped/empty SCENE_IDs from possible SCENE_ID list
        # DEADBEEF - This is modifying the master export_id list
        #   Commenting out for now
        # export_ids = export_ids - missing_skip_ids
        logging.debug('    Scenes intersecting zone: {}'.format(
            ', '.join(sorted(missing_zone_ids))))
        logging.debug('    Scenes not intersecting zone: {}'.format(
            ', '.join(sorted(missing_nonzone_ids))))
        logging.info('    Include ID count: {}'.format(
            len(missing_zone_ids)))
        logging.info('    Exclude ID count: {}'.format(
            len(missing_nonzone_ids)))

        # Create empty entries for all scenes that are missing
        # Identify whether the missing scenes intersect the zone or not
        # Set pixels counts for non-intersecting SCENE_IDs to 0
        # Set pixel counts for intersecting SCENE_IDs to NaN
        #   This will allow the zonal stats to set actual pixel count value
        logging.debug('    Appending all empty SCENE_IDs')
        missing_df = pd.DataFrame(
            index=missing_all_ids, columns=output_df.columns)
        missing_df.index.name = 'SCENE_ID'
        missing_df['ZONE_NAME'] = zone['name']
        missing_df['ZONE_FID'] = zone['fid']
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
        # DEADBEEF - Setting ROW to np.nan will cause problems in QAQ and
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

        # Set intersecting zone pixel counts to 0
        if missing_nonzone_ids:
            missing_nonzone_mask = missing_df.index.isin(missing_nonzone_ids)
            missing_df.loc[missing_nonzone_mask, 'ROW'] = 0
            missing_df.loc[missing_nonzone_mask, 'PIXEL_COUNT'] = 0
            missing_df.loc[missing_nonzone_mask, 'PIXEL_TOTAL'] = 0
            missing_df.loc[missing_nonzone_mask, 'FMASK_COUNT'] = 0
            missing_df.loc[missing_nonzone_mask, 'FMASK_TOTAL'] = 0

        # Remove the overlapping missing entries
        # Then append the new missing entries to the zone CSV
        # if zone_df.index.intersection(missing_df.index).any():
        try:
            zone_df.drop(
                zone_df.index.intersection(missing_df.index),
                inplace=True)
        except ValueError:
            pass
        zone_df = zone_df.append(missing_df, sort=False)
        logging.debug('    Writing to CSV')
        csv_writer(zone_df, zone['csv'], export_fields)

        # Update the master dataframe
        zone_df.reset_index(inplace=True)
        zone_df.set_index(
            ['ZONE_NAME', 'SCENE_ID'], inplace=True, drop=True)
        try:
            output_df.drop(
                output_df.index.get_level_values('ZONE_NAME')==zone['name'],
                inplace=True)
        except Exception as e:
            # These seem to happen with the zone is not in the output_df
            logging.debug('    Exception: {}'.format(e))
            pass
        output_df = output_df.append(zone_df, sort=False)

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

        data_df.set_index(
            ['ZONE_NAME', 'SCENE_ID'], inplace=True, drop=True)
        return data_df

    # Write values to file after each year
    # Group export dates by year (and path/row also?)
    export_ids_iter = [
        [year, list(dates)]
        for year, dates in groupby(sorted(missing_ids), lambda x: x[12:16])]
    for export_year, export_ids in export_ids_iter:
        logging.debug('\n  Iter year: {}'.format(export_year))

        # These can be mosaiced or single scene IDs depending on mosaic method
        for export_id in sorted(export_ids):
            logging.info('  {}'.format(export_id))

            if ini['INPUTS']['mosaic_method'] in landsat.mosaic_options:
                export_id_list = mosaic_id_dict[export_id]
                logging.debug('    SCENE_IDs: {}'.format(', '.join(export_id_list)))
            else:
                export_id_list = list(export_id)

            # DEADBEEF - There has to be a way to do this selection in one line
            scene_df = output_df.iloc[
                output_df.index.get_level_values('SCENE_ID')==export_id]
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
                    {'ZONE_NAME': zone['name'], 'ZONE_FID': zone['fid']})
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
                    'ZONE_FID': ee.Number(ftr.get('ZONE_FID')),
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
                'ZONE_FID': -9999,
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

            if ini['EXPORT']['export_dest'] == 'getinfo':
                logging.debug('    Requesting data')
                export_df = pd.DataFrame([
                    ftr['properties']
                    for ftr in utils.ee_getinfo(stats_coll)['features']])
                export_df = export_update(export_df)

                # Save data to main dataframe
                if not export_df.empty:
                    logging.debug('    Processing data')
                    if overwrite_flag:
                        # Update happens inplace automatically
                        output_df.update(export_df)
                        # output_df = output_df.append(export_df, sort=False)
                    else:
                        # Combine first doesn't have an inplace parameter
                        output_df = output_df.combine_first(export_df)

        # Save updated CSV
        # if output_df is not None and not output_df.empty:
        if not output_df.empty:
            logging.info('\n  Writing zone CSVs')
            for zone in zones:
                logging.debug(
                    '  ZONE: {} (FID: {})'.format(zone['name'], zone['fid']))
                # logging.debug('    {}'.format(zone_output_path))
                zone_df = output_df.iloc[
                    output_df.index.get_level_values('ZONE_NAME')==zone['name']]
                csv_writer(zone_df, zone['csv'], export_fields)
        else:
            logging.info('\n  Empty output dataframe')


def gridmet_daily_func(export_fields, ini, zones_geojson, zones_wkt,
                       gridmet_end_dt, overwrite_flag=False):
    """

    Parameters
    ----------
    export_fields : list
    ini : dict
        Input file parameters.
    zones_geojson : dict
        Zone specific parameters.
    zones_wkt : str
        Zones spatial reference Well Known Text.
    gridmet_end_dt : datetime
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    """

    logging.info('\nGRIDMET Daily ETo/PPT')

    # DEADBEEF - For now, hardcode transform to a standard Landsat image
    ini['EXPORT']['transform'] = [30.0, 0.0, 15.0, 0.0, -30.0, 15.0]
    # ini['EXPORT']['transform'] = '[{}]'.format(','.join(
    #     map(str, 30.0, 0.0, 15.0, 0.0, -30.0, 15.0)))
    logging.debug('  Output Transform: {}'.format(
        ini['EXPORT']['transform']))

    gridmet_products = ini['ZONAL_STATS']['gridmet_products'][:]
    gridmet_fields = [f.upper() for f in gridmet_products]

    def csv_writer(output_df, output_path, output_fields):
        """Write the dataframe to CSV with custom formatting"""
        csv_df = output_df.copy()

        # Convert float fields to objects, set NaN to None
        for field in csv_df.columns.values:
            if field.upper() not in gridmet_fields:
                continue
            csv_df[field] = csv_df[field].astype(object)
            null_mask = csv_df[field].isnull()
            csv_df.loc[null_mask, field] = None
            csv_df.loc[~null_mask, field] = csv_df.loc[
                ~null_mask, field].map(
                lambda x: '{0:10.6f}'.format(x).strip())

        # Set field types
        for field in ['ZONE_FID', 'YEAR', 'MONTH', 'WATER_YEAR']:
            csv_df[field] = csv_df[field].astype(int)
        # if csv_df['ZONE_NAME'].dtype == np.float64:
        #     csv_df['ZONE_NAME'] = csv_df['ZONE_NAME'].astype(int).astype(str)

        csv_df.reset_index(drop=False, inplace=True)
        csv_df.sort_values(by=['DATE'], inplace=True)
        csv_df.to_csv(output_path, index=False,
                      columns=output_fields)

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
        output_df = pd.concat(zone_df_list, sort=False)
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

    # Get list of possible dates based on INI
    export_dates = set(
        date_str for date_str in utils.date_range(
            '{}-01-01'.format(ini['INPUTS']['start_year'] - 1),
            '{}-12-31'.format(ini['INPUTS']['end_year']))
        if datetime.datetime.strptime(date_str, '%Y-%m-%d') <= gridmet_end_dt)
    # logging.debug('  Export Dates: {}'.format(
    #     ', '.join(sorted(export_dates))))

    # For overwrite, drop all expected entries from existing output DF
    if overwrite_flag:
        output_df = output_df[
            ~output_df.index.get_level_values('DATE').isin(list(export_dates))]

    # Identify missing dates in any zone
    # Iterate by zone in order to set zone name/fid
    logging.debug('\n  Identifying dates/zones with missing data')
    missing_dates = set()
    for zone_name, zone_fid, zone_output_path in zone_list:
        logging.debug('  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
        zone_df = output_df.iloc[
            output_df.index.get_level_values('ZONE_NAME') == zone_name]
        zone_df.reset_index(inplace=True)
        zone_df.set_index(['DATE'], inplace=True, drop=True)

        # Get list of existing dates in the CSV
        if not zone_df.empty:
            zone_dates = set(zone_df.index.values)
        else:
            zone_dates = set()

        missing_all_dates = export_dates - zone_dates
        missing_dates.update(missing_all_dates)
        # logging.debug('    Dates missing all values: {}'.format(
        #     ', '.join(sorted(missing_all_dates))))

        # Add non-intersecting SCENE_IDs directly to the output dataframe
        if missing_all_dates:
            logging.debug('    Appending missing dates')
            missing_df = pd.DataFrame(
                index=missing_all_dates, columns=output_df.columns)
            missing_index = pd.to_datetime(missing_df.index, format='%Y-%m-%d')
            missing_df.index.name = 'DATE'
            missing_df['ZONE_NAME'] = zone_name
            missing_df['ZONE_FID'] = zone_fid
            missing_df['YEAR'] = missing_index.year
            missing_df['MONTH'] = missing_index.month
            missing_df['DAY'] = missing_index.day
            missing_df['DOY'] = missing_index.dayofyear.astype(int)
            # Build the datetime for the start of the month
            # Move the datetime forward 3 months
            # Get the year
            missing_df['WATER_YEAR'] = (
                pd.to_datetime(
                    missing_index.strftime('%Y-%m-01'), format='%Y-%m-%d') +
                pd.DateOffset(months=3)).year

            # Remove the overlapping missing entries
            # Then append the new missing entries to the zone CSV
            # if zone_df.index.intersection(missing_df.index).any():
            try:
                zone_df.drop(
                    zone_df.index.intersection(missing_df.index), inplace=True)
            except ValueError:
                pass
            zone_df = zone_df.append(missing_df, sort=False)
            csv_writer(zone_df, zone_output_path, export_fields)

            # Update the master dataframe
            zone_df.reset_index(inplace=True)
            zone_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)
            try:
                output_df.drop(
                    output_df.index.get_level_values('ZONE_NAME') == zone_name,
                    inplace=True)
            except (ValueError, KeyError):
                # These seem to happen with the zone is not in the output_df
                pass
            output_df = output_df.append(zone_df, sort=False)

    # Identify SCENE_IDs that are missing any data
    # Filter based on product and SCENE_ID lists
    missing_date_mask = output_df.index.get_level_values('DATE') \
        .isin(export_dates)
    missing_df = output_df.loc[missing_date_mask, gridmet_fields].isnull()

    # List of SCENE_IDs and products with some missing data
    missing_any_dates = set(missing_df[
        missing_df.any(axis=1)].index.get_level_values('DATE').values)
    # logging.debug('\n  Dates missing any values: {}'.format(
    #     ', '.join(sorted(missing_any_dates))))

    missing_dates.update(missing_any_dates)
    # logging.debug('\n  Dates missing values: {}'.format(
    #     ', '.join(sorted(missing_dates))))

    # Skip processing if all dates already exist in the CSV
    if not missing_dates and not overwrite_flag:
        logging.info('\n  All dates present, returning')
        return True
    export_date_list = sorted(missing_dates)

    # Update/limit GRIDMET products list if necessary
    if not missing_df.empty:
        gridmet_products = set(
            f.lower()
            for f in missing_df.columns[missing_df.any(axis=0)]
            if f.lower() in gridmet_products)
        logging.debug('\n  Products missing any values: {}'.format(
            ', '.join(sorted(gridmet_products))))

    # Identify zones with missing data
    missing_zones = set(
        missing_df[missing_df.any(axis=1)].index.get_level_values('ZONE_NAME'))
    logging.debug('\n  Zones with missing data: {}'.format(
        ', '.join(sorted(missing_zones))))

    # Build collection of all features to test for each SCENE_ID
    # I have to build a geometry in order to set a non-WGS84 projection
    # Limit zones collection to only zones with missing data
    zone_ftr_list = []
    for z in zones_geojson['features']:
        zone_name = str(z['properties'][ini['INPUTS']['zone_field']]) \
            .replace(' ', '_')
        if zone_name not in missing_zones:
            continue
        zone_ftr_list.append(ee.Feature(
            ee.Geometry(
                geo_json=z['geometry'], opt_proj=zones_wkt, opt_geodesic=False),
            {
                'ZONE_NAME': zone_name,
                'ZONE_FID': int(z['id'])
            }))
    zone_coll = ee.FeatureCollection(zone_ftr_list)

    # Write values to file after each year
    # Group export dates by year
    export_dates_iter = [
        [year, list(dates)]
        for year, dates in groupby(sorted(missing_dates), lambda x: x[:4])]
    for export_year, export_dates in export_dates_iter:
        logging.debug('\n  Iter year: {}'.format(export_year))
        # logging.debug('  Iter dates: {}'.format(
        #     ', '.join(sorted(export_dates))))

        for export_date in sorted(export_dates):
            export_dt = datetime.datetime.strptime(export_date, '%Y-%m-%d')
            logging.info('  {}'.format(export_date))

            # Map over features for one image
            gridmet_image = ee.Image('IDAHO_EPSCOR/GRIDMET/{}'.format(
                export_dt.strftime('%Y%m%d')))
            image = gridmet_image.select(['eto', 'pr'], ['eto', 'ppt'])

            # Calculate values and statistics
            def gridmet_daily_zs_func(ftr):
                """"""
                date = ee.Date(image.get('system:time_start'))
                year = ee.Number(date.get('year'))
                month = ee.Number(date.get('month'))
                wyear = ee.Number(ee.Date.fromYMD(
                    year, month, 1).advance(3, 'month').get('year'))
                input_mean = ee.Image(image) \
                    .reduceRegion(
                        ee.Reducer.mean(),
                        geometry=ftr.geometry(),
                        crs=ini['SPATIAL']['crs'],
                        crsTransform=ini['EXPORT']['transform'],
                        bestEffort=False,
                        tileScale=1)
                        # maxPixels=zone['max_pixels'] * 3)

                # Standard output
                zs_dict = {
                    'ZONE_NAME': ee.String(ftr.get('ZONE_NAME')),
                    'ZONE_FID': ee.Number(ftr.get('ZONE_FID')),
                    'DATE': date.format('YYYY-MM-dd'),
                    'YEAR': year,
                    'MONTH': month,
                    'DAY': ee.Number(date.get('day')),
                    'DOY': ee.Number(date.getRelative('day', 'year')).add(1),
                    'WATER_YEAR': wyear
                }

                # Product specific output
                zs_dict.update({
                    p.upper(): input_mean.get(p.lower())
                    for p in gridmet_products
                })
                return ee.Feature(None, zs_dict)

            stats_coll = zone_coll.map(gridmet_daily_zs_func)

            # DEADBEEF - Only allowing getInfo calls for zonal stats by image
            if ini['EXPORT']['export_dest'] == 'getinfo':
                # logging.debug('    Requesting data')
                export_df = pd.DataFrame([
                    ftr['properties']
                    for ftr in utils.ee_getinfo(stats_coll)['features']])

                # Save data to main dataframe
                if not export_df.empty:
                    # logging.debug('    Processing data')
                    export_df.set_index(
                        ['ZONE_NAME', 'DATE'], inplace=True, drop=True)
                    if overwrite_flag:
                        # Update happens inplace automatically
                        output_df.update(export_df)
                        # output_df = output_df.append(export_df, sort=False)
                    else:
                        # Combine first doesn't have an inplace parameter
                        output_df = output_df.combine_first(export_df)

                        # Save updated CSVs
        if not output_df.empty:
            logging.info('\n  Writing zone CSVs')
            for zone_name, zone_fid, zone_output_path in zone_list:
                logging.debug(
                    '  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
                # logging.debug('    {}'.format(zone_output_path))
                if zone_name not in missing_zones:
                    logging.debug('    No missing values, skipping zone')
                    continue

                zone_df = output_df.iloc[
                    output_df.index.get_level_values('ZONE_NAME') == zone_name]
                zone_df.reset_index(inplace=True)
                zone_df.set_index(['DATE'], inplace=True, drop=True)
                if zone_df.empty:
                    logging.debug('    Empty zone df, skipping')
                    continue
                csv_writer(zone_df, zone_output_path, export_fields)


def gridmet_monthly_func(export_fields, ini, zones_geojson, zones_wkt,
                         gridmet_end_dt, overwrite_flag=False):
    """

    Parameters
    ----------
    export_fields : list
    ini : dict
        Input file parameters.
    zones_geojson : dict
        Zone specific parameters.
    zones_wkt : str
        Zones spatial reference Well Known Text.
    gridmet_end_dt : datetime
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    """

    logging.info('\nGRIDMET Monthly ETo/PPT')

    # DEADBEEF - For now, hardcode transform to a standard Landsat image
    ini['EXPORT']['transform'] = [30.0, 0.0, 15.0, 0.0, -30.0, 15.0]
    # ini['EXPORT']['transform'] = '[{}]'.format(','.join(
    #     map(str, 30.0, 0.0, 15.0, 0.0, -30.0, 15.0)))
    logging.debug('  Output Transform: {}'.format(
        ini['EXPORT']['transform']))

    gridmet_products = ini['ZONAL_STATS']['gridmet_products'][:]
    gridmet_fields = [f.upper() for f in gridmet_products]

    def csv_writer(output_df, output_path, output_fields):
        """Write the dataframe to CSV with custom formatting"""
        csv_df = output_df.copy()

        # Convert float fields to objects, set NaN to None
        for field in csv_df.columns.values:
            if field.upper() not in gridmet_fields:
                continue
            csv_df[field] = csv_df[field].astype(object)
            null_mask = csv_df[field].isnull()
            csv_df.loc[null_mask, field] = None
            csv_df.loc[~null_mask, field] = csv_df.loc[
                ~null_mask, field].map(
                lambda x: '{0:10.6f}'.format(x).strip())

        # Set field types
        for field in ['ZONE_FID', 'YEAR', 'MONTH', 'WATER_YEAR']:
            csv_df[field] = csv_df[field].astype(int)
        # if csv_df['ZONE_NAME'].dtype == np.float64:
        #     csv_df['ZONE_NAME'] = csv_df['ZONE_NAME'].astype(int).astype(str)

        csv_df.reset_index(drop=False, inplace=True)
        csv_df.sort_values(by=['DATE'], inplace=True)
        csv_df.to_csv(output_path, index=False,
                      columns=output_fields)

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
        output_df = pd.concat(zone_df_list, sort=False)
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

    # Get list of possible dates based on INI
    export_dates = set([
        datetime.datetime(y, m, 1).strftime('%Y-%m-%d')
        for y in range(
            ini['INPUTS']['start_year'] - 1, ini['INPUTS']['end_year'] + 1)
    for m in range(1, 13)
        if datetime.datetime(y, m, 1) <= gridmet_end_dt])
    # logging.debug('  Export Dates: {}'.format(
    #     ', '.join(sorted(export_dates))))

    # For overwrite, drop all expected entries from existing output DF
    if overwrite_flag:
        output_df = output_df[
            ~output_df.index.get_level_values('DATE').isin(list(export_dates))]

    # Identify missing dates in any zone
    # Iterate by zone in order to set zone name/fid
    logging.debug('\n  Identifying dates/zones with missing data')
    missing_dates = set()
    for zone_name, zone_fid, zone_output_path in zone_list:
        logging.debug('  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
        zone_df = output_df.iloc[
            output_df.index.get_level_values('ZONE_NAME') == zone_name]
        zone_df.reset_index(inplace=True)
        zone_df.set_index(['DATE'], inplace=True, drop=True)

        # Get list of existing dates in the CSV
        if not zone_df.empty:
            zone_dates = set(zone_df.index.values)
        else:
            zone_dates = set()

        missing_all_dates = export_dates - zone_dates
        missing_dates.update(missing_all_dates)
        # logging.debug('    Dates missing all values: {}'.format(
        #     ', '.join(sorted(missing_all_dates))))

        # Add non-intersecting SCENE_IDs directly to the output dataframe
        if missing_all_dates:
            logging.debug('    Appending missing dates')
            missing_df = pd.DataFrame(
                index=missing_all_dates, columns=output_df.columns)
            missing_index = pd.to_datetime(missing_df.index, format='%Y-%m-%d')
            missing_df.index.name = 'DATE'
            missing_df['ZONE_NAME'] = zone_name
            missing_df['ZONE_FID'] = zone_fid
            missing_df['YEAR'] = missing_index.year
            missing_df['MONTH'] = missing_index.month
            # missing_df['DAY'] = missing_index.day
            # missing_df['DOY'] = missing_index.dayofyear.astype(int)
            # Build the datetime for the start of the month
            # Move the datetime forward 3 months
            # Get the year
            missing_df['WATER_YEAR'] = (
                pd.to_datetime(
                    missing_index.strftime('%Y-%m-01'), format='%Y-%m-%d') +
                pd.DateOffset(months=3)).year

            # Remove the overlapping missing entries
            # Then append the new missing entries to the zone CSV
            # if zone_df.index.intersection(missing_df.index).any():
            try:
                zone_df.drop(
                    zone_df.index.intersection(missing_df.index), inplace=True)
            except ValueError:
                pass
            zone_df = zone_df.append(missing_df, sort=False)
            csv_writer(zone_df, zone_output_path, export_fields)

            # Update the master dataframe
            zone_df.reset_index(inplace=True)
            zone_df.set_index(['ZONE_NAME', 'DATE'], inplace=True, drop=True)
            try:
                output_df.drop(
                    output_df.index.get_level_values('ZONE_NAME') == zone_name,
                    inplace=True)
            except (ValueError, KeyError):
                # These seem to happen with the zone is not in the output_df
                pass
            output_df = output_df.append(zone_df, sort=False)

    # Identify SCENE_IDs that are missing any data
    # Filter based on product and SCENE_ID lists
    missing_date_mask = output_df.index.get_level_values('DATE')\
        .isin(export_dates)
    missing_df = output_df.loc[missing_date_mask, gridmet_fields].isnull()

    # List of SCENE_IDs and products with some missing data
    missing_any_dates = set(missing_df[
        missing_df.any(axis=1)].index.get_level_values('DATE').values)
    # logging.debug('\n  Dates missing any values: {}'.format(
    #     ', '.join(sorted(missing_any_dates))))

    missing_dates.update(missing_any_dates)
    logging.debug('\n  Dates missing values: {}'.format(
        ', '.join(sorted(missing_dates))))

    # Skip processing if all dates already exist in the CSV
    if not missing_dates and not overwrite_flag:
        logging.info('\n  All dates present, returning')
        return True
    export_date_list = sorted(missing_dates)

    # Update/limit GRIDMET products list if necessary
    if not missing_df.empty:
        gridmet_products = set(
            f.lower()
            for f in missing_df.columns[missing_df.any(axis=0)]
            if f.lower() in gridmet_products)
        logging.debug('\n  Products missing any values: {}'.format(
            ', '.join(sorted(gridmet_products))))

    # Identify zones with missing data
    missing_zones = set(
        missing_df[missing_df.any(axis=1)].index.get_level_values('ZONE_NAME'))
    logging.debug('\n  Zones with missing data: {}'.format(
        ', '.join(sorted(missing_zones))))

    # Build collection of all features to test for each SCENE_ID
    # I have to build a geometry in order to set a non-WGS84 projection
    # Limit zones collection to only zones with missing data
    zone_ftr_list = []
    for z in zones_geojson['features']:
        zone_name = str(z['properties'][ini['INPUTS']['zone_field']])\
            .replace(' ', '_')
        if zone_name not in missing_zones:
            continue
        zone_ftr_list.append(ee.Feature(
            ee.Geometry(
                geo_json=z['geometry'], opt_proj=zones_wkt, opt_geodesic=False),
            {
                'ZONE_NAME': zone_name,
                'ZONE_FID': int(z['id'])
            }))
    zone_coll = ee.FeatureCollection(zone_ftr_list)

    logging.debug('\n  Computing zonal stats')
    for export_date in export_date_list:
        logging.info('  {}'.format(export_date))

        # Compute monthly sums of GRIDMET
        # def monthly_sum(start_dt):
        #     gridmet_coll = ee.ImageCollection(
        #         ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
        #             .select(['eto', 'pr']) \
        #             .filterDate(
        #                 ee.Date(start_dt),
        #                 ee.Date(start_dt).advance(1, 'month')))
        #     return ee.Image(gridmet_coll.sum()) \
        #         .select([0, 1], ['eto', 'ppt']) \
        #         .set('system:time_start', ee.Date(start_dt).millis())

        # Compute monthly sums of GRIDMET
        def gridmet_monthly(start_dt):
            gridmet = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate(
                ee.Date(start_dt),
                ee.Date(start_dt).advance(1, 'month'))

            def image_mean(image):
                return ee.Image(image.reduce(ee.Reducer.mean()))

            # Sum depth units
            output_images = []
            if 'ppt' in gridmet_products:
                output_images.append(ee.Image(
                    gridmet.select(['pr'], ['ppt']).sum()))
            if 'eto' in gridmet_products:
                output_images.append(ee.Image(gridmet.select(['eto']).sum()))

            # Average other units
            if 'tmin' in gridmet_products:
                output_images.append(ee.Image(
                    gridmet.select(['tmmn'], ['tmin']).mean()))
            if 'tmax' in gridmet_products:
                output_images.append(ee.Image(
                    gridmet.select(['tmmx'], ['tmax']).mean()))
            if 'tmean' in gridmet_products:
                output_images.append(ee.Image(gridmet.select(['tmmn', 'tmmx']) \
                    .map(image_mean).mean()).rename(['tmean']))

            return ee.Image(output_images) \
                .set('system:time_start', ee.Date(start_dt).millis())

        # Map over features for one image
        image = gridmet_monthly(export_date)

        # Calculate values and statistics
        # Build function in loop to set water year ETo/PPT values
        def gridmet_monthly_zs_func(ftr):
            """"""
            date = ee.Date(image.get('system:time_start'))
            year = ee.Number(date.get('year'))
            month = ee.Number(date.get('month'))
            wyear = ee.Number(ee.Date.fromYMD(
                year, month, 1).advance(3, 'month').get('year'))
            input_mean = ee.Image(image) \
                .reduceRegion(
                    ee.Reducer.mean(),
                    geometry=ftr.geometry(),
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=ini['EXPORT']['transform'],
                    bestEffort=False,
                    tileScale=1)
                    # maxPixels=zone['max_pixels'] * 3)

            # Standard output
            zs_dict = {
                'ZONE_NAME': ee.String(ftr.get('ZONE_NAME')),
                'ZONE_FID': ee.Number(ftr.get('ZONE_FID')),
                'DATE': date.format('YYYY-MM-dd'),
                'YEAR': year,
                'MONTH': month,
                'WATER_YEAR': wyear
            }

            # Product specific output
            zs_dict.update({
                p.upper(): input_mean.get(p.lower())
                for p in gridmet_products
            })
            return ee.Feature(None, zs_dict)

        stats_coll = zone_coll.map(gridmet_monthly_zs_func)

        # DEADBEEF - Only allowing getInfo calls for zonal stats by image
        if ini['EXPORT']['export_dest'] == 'getinfo':
            # logging.debug('    Requesting data')
            export_df = pd.DataFrame([
                ftr['properties']
                for ftr in utils.ee_getinfo(stats_coll)['features']])

            # Save data to main dataframe
            if not export_df.empty:
                # logging.debug('    Processing data')
                export_df.set_index(
                    ['ZONE_NAME', 'DATE'], inplace=True, drop=True)
                if overwrite_flag:
                    # Update happens inplace automatically
                    output_df.update(export_df)
                    # output_df = output_df.append(export_df, sort=False)
                else:
                    # Combine first doesn't have an inplace parameter
                    output_df = output_df.combine_first(export_df)

    # Save updated CSVs
    if not output_df.empty:
        logging.info('\n  Writing zone CSVs')
        for zone_name, zone_fid, zone_output_path in zone_list:
            logging.debug('  ZONE: {} (FID: {})'.format(zone_name, zone_fid))
            # logging.debug('    {}'.format(zone_output_path))
            if zone_name not in missing_zones:
                logging.debug('    No missing values, skipping zone')
                continue

            # zone_df = output_df[output_df['ZONE_NAME']==zone_name]
            zone_df = output_df.iloc[
                output_df.index.get_level_values('ZONE_NAME') == zone_name]
            zone_df.reset_index(inplace=True)
            zone_df.set_index(['DATE'], inplace=True, drop=True)
            if zone_df.empty:
                logging.debug('    Empty zone df, skipping')
                continue
            csv_writer(zone_df, zone_output_path, export_fields)


# def pdsi_func(export_fields, ini, zones, tasks, overwrite_flag=False):
#     """
#
#     Parameters
#     ----------
#     export_fields : list
#     ini : dict
#         Input file parameters.
#     zone : dict
#         Zone specific parameters.
#     tasks :
#     overwrite_flag : bool, optional
#         If True, overwrite existing files (the default is False).
#
#     """
#
#     logging.info('  GRIDMET PDSI')
#
#     pdsi_coll = ee.ImageCollection('IDAHO_EPSCOR/PDSI') \
#         .select(['pdsi'], ['pdsi']) \
#         .filterDate(
#             '{}-01-01'.format(ini['INPUTS']['start_year']),
#             '{}-01-01'.format(ini['INPUTS']['end_year'] + 1))
#     export_id = '{}_{}_pdsi_dekad'.format(
#         os.path.splitext(ini['INPUTS']['zone_filename'])[0],
#         zone['name'].lower())
#     output_id = '{}_pdsi_dekad'.format(zone['name'])
#
#     export_path = os.path.join(
#         ini['EXPORT']['export_ws'], export_id + '.csv')
#     output_path = os.path.join(zone['output_ws'], output_id + '.csv')
#     logging.debug('    Export: {}'.format(export_id + '.csv'))
#     logging.debug('    Output: {}'.format(output_path))
#
#     # There is an EE bug that appends "ee_export" to the end of CSV
#     #   file names when exporting to cloud storage
#     # Also, use the sharelink path for reading the csv directly
#     if ini['EXPORT']['export_dest'] == 'cloud':
#         export_cloud_name = export_id + 'ee_export.csv'
#         export_cloud_path = os.path.join(
#             ini['EXPORT']['export_ws'], export_cloud_name)
#         export_cloud_url = 'https://storage.googleapis.com/{}/{}'.format(
#             ini['EXPORT']['bucket_name'], export_cloud_name)
#
#     if overwrite_flag:
#         if export_id in tasks.keys():
#             logging.debug('  Task already submitted, cancelling')
#             ee.data.cancelTask(tasks[export_id])
#             del tasks[export_id]
#
#         if (ini['EXPORT']['export_dest'] == 'gdrive' and
#                 os.path.isfile(export_path)):
#             logging.debug('  Export CSV already exists, removing')
#             os.remove(export_path)
#         elif (ini['EXPORT']['export_dest'] == 'cloud' and
#                 export_cloud_name in ini['EXPORT']['file_list']):
#             logging.debug('    Export image already exists')
#             # # Files in cloud storage are easily overwritten
#             # #   so it is unneccesary to manually remove them
#             # # This would remove an existing file
#             # check_output(['gsutil', 'rm', export_path])
#
#         if os.path.isfile(output_path):
#             logging.debug('    Output CSV already exists, removing')
#             os.remove(output_path)
#
#     # This should probably be moved into an else block
#     #   to avoid lots of os.path.isfile calls when overwriting
#     if export_id in tasks.keys():
#         logging.debug('  Task already submitted, skipping')
#         return True
#     elif (ini['EXPORT']['export_dest'] == 'gdrive' and
#             os.path.isfile(export_path)):
#         logging.debug('  Export CSV already exists, moving')
#         # Modify CSV while copying from Google Drive
#         try:
#             export_df = pd.read_csv(export_path)
#             export_df = export_df[export_fields]
#             export_df.sort_values(by=['DATE'], inplace=True)
#             export_df.to_csv(
#                 output_path, index=False, columns=export_fields)
#         except pd.io.common.EmptyDataError:
#             # Save an empty dataframe to the output path
#             logging.warning('    Empty dataframe')
#             export_df = pd.DataFrame(columns=export_fields)
#             export_df.to_csv(
#                 output_path, index=False, columns=export_fields)
#             # logging.warning('    Empty dataframe, skipping')
#             # continue
#         os.remove(export_path)
#         return True
#     elif (ini['EXPORT']['export_dest'] == 'cloud' and
#             export_cloud_name in ini['EXPORT']['cloud_file_list']):
#         logging.debug('    Export file already exists, moving')
#         logging.debug('    Reading {}'.format(export_cloud_url))
#         try:
#             export_request = requests.get(export_cloud_url).content
#             export_df = pd.read_csv(
#                 StringIO(export_request.decode('utf-8')))
#             export_df = export_df[export_fields]
#             export_df.sort_values(by=['DATE'], inplace=True)
#             export_df.to_csv(
#                 output_path, index=False, columns=export_fields)
#         except pd.io.common.EmptyDataError:
#             # Save an empty dataframe to the output path
#             logging.warning('    Empty dataframe')
#             export_df = pd.DataFrame(columns=export_fields)
#             export_df.to_csv(
#                 output_path, index=False, columns=export_fields)
#             # logging.warning('    Empty dataframe, skipping')
#             # continue
#         except Exception as e:
#             logging.error('Unhandled Exception')
#             logging.error(str(e))
#             return False
#         logging.debug('    Removing {}'.format(export_cloud_path))
#         try:
#             check_output(['gsutil', 'rm', export_cloud_path])
#         except Exception as e:
#             logging.error('Unhandled Exception')
#             logging.error(str(e))
#         return True
#     elif os.path.isfile(output_path):
#         logging.debug('    Output CSV already exists, skipping')
#         return True
#
#     # Calculate values and statistics
#     # Build function in loop to set water year ETo/PPT values
#     def pdsi_zonal_stats_func(image):
#         """"""
#         date = ee.Date(image.get('system:time_start'))
#         doy = ee.Number(date.getRelative('day', 'year')).add(1)
#         input_mean = ee.Image(image) \
#             .reduceRegion(
#                 ee.Reducer.mean(), geometry=zone['geom'],
#                 crs=ini['SPATIAL']['crs'],
#                 crsTransform=ini['EXPORT']['transform'],
#                 bestEffort=False, tileScale=1,
#                 maxPixels=zone['max_pixels'] * 2)
#         return ee.Feature(
#             None,
#             {
#                 'ZONE_NAME': zone['name'],
#                 'ZONE_FID': zone['fid'],
#                 'DATE': date.format('YYYY-MM-dd'),
#                 'YEAR': date.get('year'),
#                 'MONTH': date.get('month'),
#                 'DAY': date.get('day'),
#                 'DOY': doy,
#                 'PDSI': input_mean.get('pdsi'),
#             })
#     stats_coll = pdsi_coll.map(pdsi_zonal_stats_func)
#
#     logging.debug('  Building export task')
#     if ini['EXPORT']['export_dest'] == 'gdrive':
#         task = ee.batch.Export.table.toDrive(
#             collection=stats_coll,
#             description=export_id,
#             folder=ini['EXPORT']['export_folder'],
#             fileNamePrefix=export_id,
#             fileFormat='CSV')
#     elif ini['EXPORT']['export_dest'] == 'cloud':
#         task = ee.batch.Export.table.toCloudStorage(
#             collection=stats_coll,
#             description=export_id,
#             bucket=ini['EXPORT']['bucket_name'],
#             fileNamePrefix='{}'.format(export_id.replace('-', '')),
#             # fileNamePrefix=export_id,
#             fileFormat='CSV')
#
#     # Download the CSV to your Google Drive
#     logging.debug('    Starting export task')
#     utils.ee_request(task.start())


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
    logging.getLogger('googleapiclient').setLevel(logging.ERROR)
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, overwrite_flag=args.overwrite)
