#--------------------------------
# Name:         ee_landsat_image_download.py
# Purpose:      Earth Engine Landsat Image Download
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
import json
import logging
import os
import subprocess
import sys

import ee
import numpy as np
from osgeo import ogr

import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils


def ee_image_download(ini_path=None, overwrite_flag=False):
    """Earth Engine Annual Mean Image Download

    Parameters
    ----------
    ini_path : str
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    """
    logging.info('\nEarth Engine Landsat Image Download')
    images_folder = 'landsat'

    if overwrite_flag:
        logging.warning(
            '\nAre you sure you want to overwrite existing images?')
        input('Press ENTER to continue')

    # Regular expression to pull out Landsat scene_id
    # landsat_re = re.compile(
    #     'L[ETC][4578]\d{6}(?P<YEAR>\d{4})(?P<DOY>\d{3})\D{3}\d{2}')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='IMAGES')

    nodata_value = -9999

    # Float32/Float64
    float_output_type = 'Float32'
    float_nodata_value = np.finfo(np.float32).min
    # Byte/Int16/UInt16/UInt32/Int32
    int_output_type = 'Byte'
    int_nodata_value = 255
    int_bands = ['cloud_score', 'fmask']

    # Get ee features from shapefile
    zone_geom_list = gdc.shapefile_2_geom_list_func(
        ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
        reverse_flag=False)

    # Check if the zone_names are unique
    # Eventually support merging common zone_names
    if len(set([z[1] for z in zone_geom_list])) != len(zone_geom_list):
        logging.error(
            '\nERROR: There appear to be duplicate zone ID/name values.'
            '\n  Currently, the values in "{}" must be unique.'
            '\n  Exiting.'.format(ini['INPUTS']['zone_field']))
        return False

    # Filter features by FID before merging geometries
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

    # Need zone_shp_path projection to build EE geometries
    zone_osr = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone_proj = gdc.osr_wkt(zone_osr)
    # zone_proj = ee.Projection(zone_proj).wkt().getInfo()
    # zone_proj = zone_proj.replace('\n', '').replace(' ', '')
    # logging.debug('  Zone Projection: {}'.format(zone_proj))

    # Check that shapefile has matching spatial reference
    if not gdc.matching_spatref(zone_osr, ini['SPATIAL']['osr']):
        logging.warning('  Zone OSR:\n{}\n'.format(zone_osr))
        logging.warning('  Output OSR:\n{}\n'.format(
            ini['SPATIAL']['osr']))
        logging.warning('  Zone Proj4:   {}'.format(zone_osr.ExportToProj4()))
        logging.warning('  Output Proj4: {}'.format(
            ini['SPATIAL']['osr'].ExportToProj4()))
        logging.warning(
            '\nWARNING: \n'
            'The output and zone spatial references do not appear to match\n'
            'This will likely cause problems!')
        input('Press ENTER to continue')
    else:
        logging.debug('  Zone Projection:\n{}\n'.format(
            zone_osr.ExportToWkt()))
        logging.debug('  Output Projection:\n{}\n'.format(
            ini['SPATIAL']['osr'].ExportToWkt()))
        logging.debug('  Output Cellsize: {}'.format(
            ini['SPATIAL']['cellsize']))

    # Keyword arguments for ee_common.get_landsat_collection() and
    #   ee_common.get_landsat_image()
    # Zone geom will be updated inside the loop
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
    # landsat_args['start_date'] = start_date
    # landsat_args['end_date'] = end_date

    # For composite images, compute all components bands
    landsat_args['products'] = ini['IMAGES']['download_bands'][:]
    if 'refl_toa' in landsat_args['products']:
        landsat_args['products'].extend([
            'blue_toa', 'green_toa', 'red_toa',
            'nir_toa', 'swir1_toa', 'swir2_toa'])
        landsat_args['products'].remove('refl_toa')
    if 'refl_sur' in landsat_args['products']:
        landsat_args['products'].extend([
            'blue_sur', 'green_sur', 'red_sur',
            'nir_sur', 'swir1_sur', 'swir2_sur'])
        landsat_args['products'].remove('refl_sur')
    if 'tasseled_cap' in landsat_args['products']:
        landsat_args['products'].extend([
            'tc_green', 'tc_bright', 'tc_wet'])
        landsat_args['products'].remove('tasseled_cap')

    # Initialize Earth Engine API key
    logging.info('\nInitializing Earth Engine')
    ee.Initialize()
    utils.ee_request(ee.Number(1).getInfo())

    # Get current running tasks
    tasks = utils.get_ee_tasks()


    # Download images for each feature separately
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone_name = zone_name.replace(' ', '_')
        logging.info('ZONE: {} (FID: {})'.format(zone_name, zone_fid))

        # Build EE geometry object for zonal stats
        zone_geom = ee.Geometry(
            geo_json=zone_json, opt_proj=zone_proj, opt_geodesic=False)
        landsat_args['zone_geom'] = zone_geom
        # logging.debug('  Centroid: {}'.format(
        #     zone_geom.centroid(100).getInfo()['coordinates']))

        # Use feature geometry to build extent, transform, and shape
        zone_extent = gdc.Extent(
            ogr.CreateGeometryFromJson(json.dumps(zone_json)).GetEnvelope())
        # zone_extent = gdc.Extent(zone_geom.GetEnvelope())
        zone_extent.ymin, zone_extent.xmax = zone_extent.xmax, zone_extent.ymin
        zone_extent = zone_extent.buffer(ini['IMAGES']['image_buffer'])
        zone_extent = zone_extent.adjust_to_snap(
            'EXPAND', ini['SPATIAL']['snap_x'], ini['SPATIAL']['snap_y'],
            ini['SPATIAL']['cellsize'])
        zone_geo = zone_extent.geo(ini['SPATIAL']['cellsize'])
        zone_transform = gdc.geo_2_ee_transform(zone_geo)
        zone_transform_str = '[' + ', '.join(map(str, zone_transform)) + ']'
        zone_shape = zone_extent.shape(ini['SPATIAL']['cellsize'])
        logging.debug('  Zone Shape: {}'.format(zone_shape))
        logging.debug('  Zone Transform: {}'.format(zone_transform_str))
        logging.debug('  Zone Extent: {}'.format(zone_extent))
        # logging.debug('  Zone Geom: {}'.format(zone_geom.getInfo()))

        # output_transform = zone_transform[:]
        output_transform_str = '[' + ', '.join(map(str, zone_transform)) + ']'
        output_shape_str = '{1}x{0}'.format(*zone_shape)
        logging.debug('  Image Transform: {}'.format(output_transform_str))
        logging.debug('  Image Shape: {}'.format(output_shape_str))

        zone_images_ws = os.path.join(
            ini['IMAGES']['output_ws'], zone_name, images_folder)
        if not os.path.isdir(zone_images_ws):
            os.makedirs(zone_images_ws)

        # Move to EE common?
        def get_collection_ids(image):
            return ee.Feature(None, {'id': image.get('SCENE_ID')})

        # Get list of available Landsat images
        landsat_obj = ee_common.Landsat(landsat_args)
        scene_id_list = [
            f['properties']['id']
            for f in landsat_obj.get_collection().map(
                get_collection_ids).getInfo()['features']]

        # Get list of unique image "dates"
        # Keep scene_id components as string for set operation
        # If not mosaicing images, include path/row in set
        #   otherwise set to None
        if not ini['INPUTS']['mosaic_method']:
            scene_id_list = set([
                (image_id[12:20], image_id[0:4], image_id[5:8], image_id[8:11])
                for image_id in scene_id_list])
        else:
            scene_id_list = set([
                (image_id[12:20], image_id[0:4], None, None)
                for image_id in scene_id_list])
        logging.debug('  Scene Count: {}\n'.format(len(scene_id_list)))

        # Process each image in the collection by date
        # Leave scene_id components as strings
        for date, landsat, path, row in sorted(scene_id_list):
            scene_dt = datetime.datetime.strptime(date, '%Y%m%d')
            year = scene_dt.strftime('%Y')
            doy = scene_dt.strftime('%j')

            # If not mosaicing images, include path/row in name
            if not ini['INPUTS']['mosaic_method']:
                landsat_str = '{}{}{}'.format(landsat, path, row)
            else:
                landsat_str = '{}'.format(landsat)
            logging.info('{} {} (DOY {})'.format(
                landsat.upper(), scene_dt.strftime('%Y-%m-%d'), doy))

            zone_year_ws = os.path.join(zone_images_ws, year)
            if not os.path.isdir(zone_year_ws):
                os.makedirs(zone_year_ws)

            # Get the prepped Landsat image by ID
            landsat_image = ee.Image(landsat_obj.get_image(
                landsat, year, doy, path, row))

            # Clip using the feature geometry
            if ini['IMAGES']['clip_landsat_flag']:
                landsat_image = landsat_image.clip(zone_geom)
            else:
                landsat_image = landsat_image.clip(ee.Geometry.Rectangle(
                    list(zone_extent), ini['SPATIAL']['crs'], False))

            # DEADBEEF - Display a single image
            # ee_common.show_thumbnail(landsat_image.visualize(
            #     bands=['fmask', 'fmask', 'fmask'], min=0, max=4))
            # ee_common.show_thumbnail(landsat_image.visualize(
            #     bands=['toa_red', 'toa_green', 'toa_blue'],
            #     min=0.05, max=0.35, gamma=1.4))
            # return True

            # Set the masked values to a nodata value
            # so that the TIF can have a nodata value other than 0 set
            landsat_image = landsat_image.unmask(nodata_value, False)

            for band in ini['IMAGES']['download_bands']:
                logging.debug('  Band: {}'.format(band))

                # Rename to match naming style from getDownloadURL
                #     image_name.band.tif
                export_id = '{}_{}_{}_{}_{}'.format(
                    ini['INPUTS']['zone_filename'], date, doy,
                    landsat_str.lower(), band.lower())
                output_id = '{}_{}_{}.{}'.format(
                    date, doy, landsat_str.lower(), band)
                logging.debug('  Export: {}'.format(export_id))
                logging.debug('  Output: {}'.format(output_id))

                export_path = os.path.join(
                    ini['EXPORT']['export_ws'], export_id + '.tif')
                output_path = os.path.join(
                    zone_year_ws, output_id + '.tif')
                logging.debug('  Export: {}'.format(export_path))
                logging.debug('  Output: {}'.format(output_path))

                if overwrite_flag:
                    if export_id in tasks.keys():
                        logging.debug('  Task already submitted, cancelling')
                        ee.data.cancelTask(tasks[export_id])
                        del tasks[export_id]
                    if os.path.isfile(export_path):
                        logging.debug(
                            '  Export image already exists, removing')
                        utils.remove_file(export_path)
                        # os.remove(export_path)
                    if os.path.isfile(output_path):
                        logging.debug(
                            '  Output image already exists, removing')
                        utils.remove_file(output_path)
                        # os.remove(output_path)
                else:
                    if os.path.isfile(export_path):
                        logging.debug('  Export image already exists, moving')
                        if band in int_bands:
                            subprocess.check_output([
                                'gdalwarp',
                                '-ot', int_output_type, '-overwrite',
                                '-of', 'GTiff', '-co', 'COMPRESS=LZW',
                                '-srcnodata', str(nodata_value),
                                '-dstnodata', str(int_nodata_value),
                                export_path, output_path])
                        else:
                            subprocess.check_output([
                                'gdalwarp',
                                '-ot', float_output_type, '-overwrite',
                                '-of', 'GTiff', '-co', 'COMPRESS=LZW',
                                '-srcnodata', str(nodata_value),
                                '-dstnodata', '{:f}'.format(float_nodata_value),
                                export_path, output_path])
                        with open(os.devnull, 'w') as devnull:
                            subprocess.check_call(
                                ['gdalinfo', '-stats', output_path],
                                stdout=devnull)
                        subprocess.check_output(
                            ['gdalmanage', 'delete', export_path])
                        continue
                    elif os.path.isfile(output_path):
                        logging.debug(
                            '  Output image already exists, skipping')
                        continue
                    elif export_id in tasks.keys():
                        logging.debug(
                            '  Task already submitted, skipping')
                        continue

                # Should composites include Ts?
                if band == 'refl_toa':
                    band_list = [
                        'blue_toa', 'green_toa', 'red_toa',
                        'nir_toa', 'swir1_toa', 'swir2_toa']
                elif band == 'refl_sur':
                    band_list = [
                        'blue_sur', 'green_sur', 'red_sur',
                        'nir_sur', 'swir1_sur', 'swir2_sur']
                elif band == 'tasseled_cap':
                    band_list = ['tc_bright', 'tc_green', 'tc_wet']
                else:
                    band_list = [band]
                band_image = landsat_image.select(band_list)

                # CGM 2016-09-26 - Don't apply any cloud masks to images
                # # Apply cloud mask before exporting
                # if fmask_flag and band not in ['refl_sur', 'cloud', 'fmask']:
                #     fmask = ee.Image(landsat_image.select(['fmask']))
                #     cloud_mask = fmask.eq(2).Or(fmask.eq(3)).Or(fmask.eq(4)).Not()
                #     band_image = band_image.updateMask(cloud_mask)

                logging.debug('  Building export task')
                # if ini['EXPORT']['export_dest'] == 'gdrive':
                task = ee.batch.Export.image.toDrive(
                    band_image,
                    description=export_id,
                    folder=ini['EXPORT']['export_folder'],
                    fileNamePrefix=export_id,
                    dimensions=output_shape_str,
                    crs=ini['SPATIAL']['crs'],
                    crsTransform=output_transform_str)
                # elif ini['EXPORT']['export_dest'] == 'cloud':
                #     task = ee.batch.Export.image.toCloudStorage(
                #         band_image,
                #         description=export_id,
                #         bucket=ini['EXPORT']['export_folder'],
                #         fileNamePrefix=export_id,
                #         dimensions=output_shape_str,
                #         crs=ini['SPATIAL']['crs'],
                #         crsTransform=output_transform_str)

                logging.debug('  Starting export task')
                utils.ee_request(task.start())


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine Landsat Image Download',
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

    ee_image_download(ini_path=args.ini, overwrite_flag=args.overwrite)
