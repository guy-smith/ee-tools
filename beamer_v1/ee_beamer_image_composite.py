#--------------------------------
# Name:         ee_beamer_composite_download.py
# Purpose:      Compute and download Beamer ETg images using Earth Engine
# Created       2017-07-27
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
from collections import defaultdict
import datetime
import json
import logging
import os
import shutil
import sys
from time import sleep
import pprint
# Python 2/3 support
try:
    import urllib.request as urlrequest
except ImportError:
    import urllib as urlrequest
import zipfile

import ee
import numpy as np
from osgeo import gdal, ogr

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
import ee_tools.wrs2 as wrs2


def ee_beamer_et(ini_path=None, overwrite_flag=False):
    """Earth Engine Beamer ET Image Download

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nEarth Engine Beamer Median/Mean ETg Image Download')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='IMAGES')
    inputs.parse_section(ini, section='BEAMER')

    ini['IMAGES']['download_bands'] = [
        'etg_mean', 'etg_lci', 'etg_uci', 'etg_lpi', 'etg_upi']
    stat_list = ['median', 'mean']
    nodata_value = -9999

    # Get ee features from shapefile
    zone_geom_list = gdc.shapefile_2_geom_list_func(
        ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
        reverse_flag=False)
    # zone_count = len(zone_geom_list)
    # output_fmt = '_{0:0%sd}.csv' % str(int(math.log10(zone_count)) + 1)

    # Check if the zone_names are unique
    # Eventually support merging common zone_names
    if len(set([z[1] for z in zone_geom_list])) != len(zone_geom_list):
        logging.error(
            '\nERROR: There appear to be duplicate zone ID/name values.'
            '\n  Currently, the values in "{}" must be unique.'
            '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
        return False

    # Filter features by FID
    if ini['INPUTS']['fid_keep_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] not in ini['INPUTS']['fid_skip_list']]

    # Merge geometries
    if ini['INPUTS']['merge_geom_flag']:
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone in zone_geom_list:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone[2])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        # merge_json = json.loads(merge_mp.ExportToJson())
        zone_geom_list = [[
            0, ini['INPUTS']['zone_filename'],
            json.loads(merge_geom.ExportToJson())]]
        ini['INPUTS']['zone_field'] = ''

    # Set all zone specific parameters into a dictionary
    zone = {}

    # Need zone_shp_path projection to build EE geometries
    zone['osr'] = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone['proj'] = gdc.osr_wkt(zone['osr'])
    # zone['proj'] = ee.Projection(zone['proj']).wkt().getInfo()
    # zone['proj'] = zone['proj'].replace('\n', '').replace(' ', '')
    # logging.debug('  Zone Projection: {}'.format(zone['proj']))

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zone['osr'], ini['SPATIAL']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zone['osr']))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.warning('  Zone Proj4:   {}'.format(
            zone['osr'].ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['SPATIAL']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(
            zone['osr'].ExportToWkt()))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['SPATIAL']['cellsize']))

    # Initialize Earth Engine API key
    logging.info('\nInitializing Earth Engine')
    ee.Initialize()
    utils.ee_request(ee.Number(1).getInfo())

    # Get list of path/row strings to centroid coordinates
    if ini['INPUTS']['tile_keep_list']:
        ini['INPUTS']['tile_geom'] = [
            wrs2.tile_centroids[tile]
            for tile in ini['INPUTS']['tile_keep_list']
            if tile in wrs2.tile_centroids.keys()]
        ini['INPUTS']['tile_geom'] = ee.Geometry.MultiPoint(
            ini['INPUTS']['tile_geom'], 'EPSG:4326')
    else:
        ini['INPUTS']['tile_geom'] = None

    # Read in ETo and PPT data from file
    if (ini['BEAMER']['eto_source'] == 'file' or
            ini['BEAMER']['ppt_source'] == 'file'):
        data_array = np.genfromtxt(
            ini['BEAMER']['data_path'], delimiter=',', names=True, dtype=None)
        data_fields = data_array.dtype.names
        logging.debug('  CSV fields: {}'.format(', '.join(data_fields)))
        # DEADBEEF - Compare fields names assuming all upper case
        data_fields = [f.upper() for f in data_fields]
        eto_dict = defaultdict(dict)
        ppt_dict = defaultdict(dict)
        for row in data_array:
            z = str(row[data_fields.index(ini['BEAMER']['data_zone_field'])])
            y = int(row[data_fields.index(ini['BEAMER']['data_year_field'])])
            if ini['BEAMER']['eto_source'] == 'file':
                # DEADBEEF - Compare fields names assuming all upper case
                eto_dict[z][y] = row[data_fields.index(
                    ini['BEAMER']['data_eto_field'].upper())]
            if ini['BEAMER']['ppt_source'] == 'file':
                # DEADBEEF - Compare fields names assuming all upper case
                ppt_dict[z][y] = row[data_fields.index(
                    ini['BEAMER']['data_ppt_field'].upper())]

    # Get filtered/merged/prepped Landsat collection
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
            'path_keep_list', 'row_keep_list', 'tile_geom',
            'adjust_method', 'mosaic_method', 'refl_sur_method']}
    #pprint.pprint(landsat_args)
    #landsat_args['products'] = ['evi_sur']
    #pprint.pprint(landsat_args)
    landsat_args['products'] = ['ndvi_sur']
    landsat = ee_common.Landsat(landsat_args)


    # Download images for each feature separately
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

        # Assume all pixels in all 14+2 images could be reduced
        zone['max_pixels'] = zone['shape'][0] * zone['shape'][1]
        logging.debug('  Max Pixels: {}'.format(zone['max_pixels']))

        # Set output spatial reference
        # Eventually allow user to manually set these
        # output_crs = zone['proj']
        logging.debug('  Image Projection: {}'.format(ini['SPATIAL']['crs']))

        # output_transform = zone['transform'][:]
        output_transform = '[' + ','.join(map(str, zone['transform'])) + ']'
        output_shape = '{1}x{0}'.format(*zone['shape'])
        logging.debug('  Image Transform: {}'.format(output_transform))
        logging.debug('  Image Shape: {}'.format(output_shape))

        zone_output_ws = os.path.join(ini['IMAGES']['output_ws'], zone_name)
        if not os.path.isdir(zone_output_ws):
            os.makedirs(zone_output_ws)

        # DEADBEEF - Much of this code could probably be moved outside the
        #   stat loop.
        # To avoid making EE calls unnecessarily, the status of the output
        #   file is checked first which requires the statistic type
        for stat in stat_list:
            logging.info('  {}'.format(stat))
            # image_id = 'etg_{}_{}'.format(
            image_id = '{}_{}'.format(
                zone_name.lower().replace(' ', '_'), stat.lower())
            zip_path = os.path.join(zone_output_ws, image_id + '.zip')

            logging.debug('  Zip: {}'.format(zip_path))
            if os.path.isfile(zip_path) and overwrite_flag:
                logging.debug('    Output already exists, removing zip')
                os.remove(zip_path)
            elif os.path.isfile(zip_path) and not overwrite_flag:
                # Check that existing ZIP files can be opened
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        pass
                except Exception as e:
                    logging.warning('    Zip file error, removing')
                    logging.debug('    {}'.format(e))
                    os.remove(zip_path)

            # Pre-compute an ETo/PPT dictionary
            ee_years = ee.List.sequence(
                ini['INPUTS']['start_year'], ini['INPUTS']['end_year'], 1)

            # Get water year PPT from file
            # Convert all input data to mm to match GRIDMET data
            if ini['BEAMER']['ppt_source'] == 'file':
                wy_ppt_input = ppt_dict[zone_name]
                if ini['BEAMER']['data_ppt_units'] == 'mm':
                    pass
                elif ini['BEAMER']['data_ppt_units'] == 'in':
                    wy_ppt_input = {
                        y: x * 25.4 for y, x in wy_ppt_input.items()}
                elif ini['BEAMER']['data_ppt_units'] == 'ft':
                    wy_ppt_input = {
                        y: x * 25.4 * 12 for y, x in wy_ppt_input.items()}
            elif ini['BEAMER']['ppt_source'] == 'gridmet':
                # Compute GRIDMET water year sums
                def gridmet_wy_ppt(year):
                    wy_start_dt = ee.Date.fromYMD(
                        ee.Number(year).subtract(1), 10, 1)
                    wy_end_dt = ee.Date.fromYMD(ee.Number(year), 10, 1)
                    wy_coll = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                        .filterDate(wy_start_dt, wy_end_dt) \
                        .select(['pr'], ['ppt'])
                    wy_image = ee.Image([
                        ee.Image(ee.ImageCollection(wy_coll).sum()),
                        ee.Image.constant(ee.Number(year)).float()])
                    return wy_image.rename(['PPT', 'YEAR']) \
                        .setMulti({'system:time_start': wy_start_dt.millis()})
                wy_ppt_coll = ee.ImageCollection(ee_years.map(gridmet_wy_ppt))

                # Get GRIDMET value at centroid of geometry
                wy_ppt_input = {
                    int(x[5]): float(x[4])
                    for x in utils.ee_getinfo(wy_ppt_coll.getRegion(
                        zone['geom'].centroid(1), 500))[1:]}
                # # Calculate GRIDMET zonal mean of geometry
                # wy_ppt_input = float(utils.ee_getinfo(ee.ImageCollection(
                #     gridmet_coll.select(['pr'], ['ppt'])).reduceRegion(
                #         reducer=ee.Reducer.sum(),
                #         geometry=zone['geom'],
                #         crs=ini['SPATIAL']['crs'],
                #         crsTransform=zone['transform'],
                #         bestEffort=False,
                #         tileScale=1))['PPT']
            # elif ini['BEAMER']['ppt_source'] == 'prism':
            #     # Calculate PRISM zonal mean of geometry
            #     wy_ppt_input = float(utils.ee_getinfo(ee.ImageCollection(
            #         prism_coll.select(['ppt'], ['ppt'])).sum().reduceRegion(
            #             reducer=ee.Reducer.mean(),
            #             geometry=zone['geom'],
            #             crs=ini['SPATIAL']['crs'],
            #             crsTransform=zone['transform'],
            #             bestEffort=False,
            #             tileScale=1))['PPT'])

            # Get water year ETo read from file
            # Convert all input data to mm for Beamer Method
            if ini['BEAMER']['eto_source'] == 'file':
                wy_eto_input = eto_dict[zone_name]
                if ini['BEAMER']['data_eto_units'] == 'mm':
                    pass
                elif ini['BEAMER']['data_eto_units'] == 'in':
                    wy_eto_input = {
                        y: x * 25.4 for y, x in wy_eto_input.items()}
                elif ini['BEAMER']['data_eto_units'] == 'ft':
                    wy_eto_input = {
                        y: x * 25.4 * 12 for y, x in wy_eto_input.items()}
            # This assumes GRIMET data is in millimeters
            elif ini['BEAMER']['eto_source'] == 'gridmet':
                # Compute GRIDMET water year sums
                def gridmet_wy_eto(year):
                    wy_start_dt = ee.Date.fromYMD(
                        ee.Number(year).subtract(1), 10, 1)
                    wy_end_dt = ee.Date.fromYMD(ee.Number(year), 10, 1)
                    wy_coll = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                        .filterDate(wy_start_dt, wy_end_dt) \
                        .select(['eto'])
                    wy_image = ee.Image([
                        ee.Image(ee.ImageCollection(wy_coll).sum()),
                        ee.Image.constant(ee.Number(year)).float()])
                    return wy_image.rename(['ETO', 'YEAR']) \
                        .setMulti({'system:time_start': wy_start_dt.millis()})
                wy_eto_coll = ee.ImageCollection(ee_years.map(gridmet_wy_eto))

                # Get GRIDMET value at centroid of geometry
                wy_eto_input = {
                    int(x[5]): float(x[4])
                    for x in utils.ee_getinfo(wy_eto_coll.getRegion(
                        zone['geom'].centroid(1), 500))[1:]}
                # wy_eto_input = float(utils.ee_getinfo(ee.ImageCollection(
                #     gridmet_coll.select(['eto'])).reduceRegion(
                #         ee.Reducer.sum(),
                #         zone['geom'],
                #         crs=ini['SPATIAL']['crs'],
                #         crsTransform=zone['transform'],
                #         bestEffort=False,
                #         tileScale=1))['ETO'])
            # logging.debug('   Input ETO [{}]  PPT [{}]'.format(
            #     ini['BEAMER']['eto_units'],
            #     ini['BEAMER']['ppt_units']))
            # for year in sorted(wy_eto_output.keys()):
            #     logging.debug('      {:>10.4f} {:>10.4f}'.format(
            #         wy_eto_output[year], wy_ppt_output[year]))

            # Scale ETo & PPT
            wy_eto_input = {
                y: p * ini['BEAMER']['eto_factor']
                for y, p in wy_eto_input.items()}
            wy_ppt_input = {
                y: p * ini['BEAMER']['ppt_factor']
                for y, p in wy_ppt_input.items()}

            # Convert output units from mm
            wy_ppt_output = wy_ppt_input.copy()
            wy_eto_output = wy_eto_input.copy()
            if ini['IMAGES']['ppt_units'] == 'mm':
                pass
            elif ini['IMAGES']['ppt_units'] == 'in':
                wy_ppt_output = {y: x / 25.4 for y, x in wy_ppt_output.items()}
            elif ini['IMAGES']['ppt_units'] == 'ft':
                wy_ppt_output = {
                    y: x / (25.4 * 12) for y, x in wy_ppt_output.items()}
            if ini['IMAGES']['eto_units'] == 'mm':
                pass
            elif ini['IMAGES']['eto_units'] == 'in':
                wy_eto_output = {y: x / 25.4 for y, x in wy_eto_output.items()}
            elif ini['IMAGES']['eto_units'] == 'ft':
                wy_eto_output = {
                    y: x / (25.4 * 12) for y, x in wy_eto_output.items()}
            logging.debug('  Year  ETo [{}]  PPT [{}]'.format(
                ini['IMAGES']['eto_units'], ini['IMAGES']['ppt_units']))
            for year in sorted(wy_eto_output.keys()):
                logging.debug('  {}  {:>8.3f}  {:>8.3f}'.format(
                    year, wy_eto_output[year], wy_ppt_output[year]))

            # Build each image separately
            etg_images = []
            for year in range(
                    ini['INPUTS']['start_year'], ini['INPUTS']['end_year'] + 1):
                # Initialize the Landsat object for target zone and iteration
                #print(year)
                landsat.zone_geom = zone['geom']
                landsat.start_date = datetime.date(year, 1, 1).isoformat()
                landsat.end_date = datetime.date(year, 12, 31).isoformat()
                landsat_coll = landsat.get_collection()
                #print(landsat_coll.first().getInfo())
                #input('')
                # print(utils.ee_getinfo(landsat_coll.aggregate_histogram('SCENE_ID')))
                # input('ENTER')

                # Add water year ETo and PPT values to each image
                def eto_ppt_func(img):
                    """"""
                    return ee.Image(img).setMulti({
                        'wy_eto': wy_eto_output[year],
                        'wy_ppt': wy_ppt_output[year]
                    })
                landsat_coll = ee.ImageCollection(
                    landsat_coll.map(eto_ppt_func))
                #print(landsat_coll.getInfo())
                #input('')
                # Build each collection separately then merge
                etg_coll = ee.ImageCollection(landsat_coll.map(
                        ee_common.beamer_func)) \
                    .select(ini['IMAGES']['download_bands'])
                #print(etg_coll.getInfo())
                #input('')
                etg_images.append(ee.Image(etg_coll.mean()).set({
                    'system:index': str(year),
                    'system:time_start': ee.Date(landsat.start_date).millis()
                    # 'year': year,
                    # 'wy_eto': wy_eto_output[year],
                    # 'wy_ppt': wy_ppt_output[year]
                }))
                #print(etg_images.getInfo())
                #input('')
            # Build the collection from the images
            etg_coll = ee.ImageCollection.fromImages(etg_images)
            #print(etg_coll.getInfo())
            #input('')
            
            if stat.lower() == 'median':
                etg_image = ee.Image(etg_coll.median())
            elif stat.lower() == 'mean':
                etg_image = ee.Image(etg_coll.mean())
            else:
                logging.error('  Unsupported stat type: {}'.format(stat))
                sys.exit()
            #print(etg_image.getInfo())
            #input('enter')

            # Clip using the feature geometry
            # Set the masked values to a nodata value
            # so that the TIF can have a nodata value other than 0 set
            etg_image = etg_image \
                .clip(zone['geom']) \
                .unmask(nodata_value, False)

            # Download the image
            if not os.path.isfile(zip_path):
                for i in range(1, 10):
                    logging.debug('  Requesting URL')
                    zip_url = utils.ee_request(etg_image.getDownloadURL({
                        'name': image_id,
                        'crs': ini['SPATIAL']['crs'],
                        'crs_transform': output_transform,
                        'dimensions': output_shape
                    }))
                    logging.debug('  {}'.format(zip_url))

                    logging.info('  Downloading')
                #for i in range(1, 10):
                    try:
                        response = urlrequest.urlopen(zip_url)
                        with open(zip_path, 'wb') as output_f:
                            shutil.copyfileobj(response, output_f)
                        break
                    except Exception as e:
                        logging.info('  Resending query')
                        logging.debug('  {}'.format(e))
                        sleep(i ** 2)
                    try:
                        os.remove(zip_path)
                    except Exception as e:
                        pass
                    # Check if file size is greater than 0?
                    # if os.path.isfile(zip_path):
                    #     break

            # Try extracting the files
            try:
                logging.info('  Extracting')
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(zone_output_ws)
            except Exception as e:
                logging.warning('    Error: could not extract'.format(i))
                logging.warning('  {}'.format(e))
                try:
                    os.remove(zip_path)
                except Exception as e:
                    pass

        logging.info('\nComputing raster statistics')
        for stat in stat_list:
            logging.info('  Stat: {}'.format(stat))
            for band in ini['IMAGES']['download_bands']:
                logging.info('  {}'.format(band))
                output_path = os.path.join(
                    # zone_output_ws, 'etg_{}_{}.{}.tif'.format(
                    zone_output_ws, '{}_{}.{}.tif'.format(
                        zone_name.lower().replace(' ', '_'),
                        stat.lower(), band.lower()))
                logging.debug('  {}'.format(output_path))

                gdc.raster_path_set_nodata(output_path, nodata_value)
                raster_statistics(output_path)


def raster_statistics(raster_path):
    """"""
    def band_statistics(input_band):
        """"""
        # stats = input_band.ComputeStatistics(False)
        stats = input_band.GetStatistics(0, 1)
        # input_band.SetStatistics(*stats)
        input_band.GetHistogram(stats[0], stats[1])
        # return stats

    raster_ds = gdal.Open(raster_path, 1)
    for band_i in range(int(raster_ds.RasterCount)):
        try:
            band = raster_ds.GetRasterBand(band_i + 1)
            band_statistics(band)
        except RuntimeError:
            logging.debug('  {} - band {} - all cells nodata'.format(
                raster_path, band_i + 1))
            continue
    raster_ds = None


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine Beamer Median ETg Image Download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True, type=utils.arg_valid_file,
        help='Input file', metavar='file')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.getLogger('googleapiclient').setLevel(logging.WARNING)

    logging.info('\n{}'.format('#' * 80))
    log_f = '{0:<20s} {1}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))
    logging.info('')

    ee_beamer_et(ini_path=args.ini, overwrite_flag=args.overwrite)
