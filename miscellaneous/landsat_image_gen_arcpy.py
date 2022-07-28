import argparse
import datetime
import logging
import os
import re
import shutil
import sys
import tempfile

import arcpy
import numpy as np


def main(overwrite_flag=False):
    # Hardcoded for now, eventually read this from the INI
    workspace = r'Y:\justinh\Projects\SNWA\ee\images\spring_valley\landsat'

    year_list = range(2016, 2017)
    # year_list = [1985, 1986, 2010, 2016]

    arcpy.CheckOutExtension('Spatial')
    # arcpy.env.workspace = workspace
    arcpy.env.overwriteOutput = True
    arcpy.env.pyramid = 'PYRAMIDS 0'
    arcpy.env.compression = "LZW"
    temp_ws = tempfile.mkdtemp()
    arcpy.env.workspace = temp_ws
    arcpy.env.scratchWorkspace = temp_ws

    nodata_value = float(np.finfo(np.float32).min)

    for year_str in os.listdir(workspace):
        if not re.match('^\d{4}$', year_str):
            continue
        elif int(year_str) not in year_list:
            continue
        logging.info(year_str)

        year_ws = os.path.join(workspace, year_str)
        for refl_sur_name in os.listdir(year_ws):
            if not refl_sur_name.endswith('.refl_sur.tif'):
                continue
            logging.info(refl_sur_name)
            refl_sur_path = os.path.join(year_ws, refl_sur_name)

            blue = arcpy.sa.Raster(refl_sur_path + '/Band_1')
            green = arcpy.sa.Raster(refl_sur_path + '/Band_2')
            red = arcpy.sa.Raster(refl_sur_path + '/Band_3')
            nir = arcpy.sa.Raster(refl_sur_path + '/Band_4')
            swir1 = arcpy.sa.Raster(refl_sur_path + '/Band_5')
            swir2 = arcpy.sa.Raster(refl_sur_path + '/Band_6')

            # NDVI
            output_path = os.path.join(
                year_ws, refl_sur_path.replace('.refl_sur.tif', '.ndvi_sur.tif'))
            if overwrite_flag and os.path.isfile(output_path):
                try:
                    arcpy.Delete_management(output_path)
                except:
                    logging.info('  Error deleting existing NDVI raster')
                    continue
            if not arcpy.Exists(output_path):
                logging.debug('  Computing NDVI')
                try:
                    output_obj = (nir - red) / (nir + red)
                    # ndvi_obj.save(output_path)
                    arcpy.CopyRaster_management(
                        output_obj, output_path, '#', '#',
                        nodata_value, '#', '#', '32_BIT_FLOAT')
                    # logging.debug('  {}'.format(output_obj))
                    # del output_obj
                except:
                    logging.info('  Error computing NDVI')
                    arcpy.Delete_management(output_path)

            # EVI
            output_path = os.path.join(
                year_ws, refl_sur_path.replace('.refl_sur.tif', '.evi_sur.tif'))
            if overwrite_flag and os.path.isfile(output_path):
                try:
                    arcpy.Delete_management(output_path)
                except:
                    logging.info('  Error deleting existing EVI raster')
                    continue
            if not arcpy.Exists(output_path):
                logging.debug('  Computing EVI')
                try:
                    output_obj = arcpy.sa.Divide(
                        2.5 * (nir - red), (nir + 6 * red - 7.5 * blue + 1))
                    arcpy.CopyRaster_management(
                        output_obj, output_path, '#', '#',
                        nodata_value, '#', '#', '32_BIT_FLOAT')
                    # logging.debug('  {}'.format(output_obj))
                    # del output_obj
                except:
                    logging.info('  Error computing EVI')
                    arcpy.Delete_management(output_path)

            # Albedo
            output_path = os.path.join(
                year_ws, refl_sur_path.replace('.refl_sur.tif', '.albedo_sur.tif'))
            if overwrite_flag and os.path.isfile(output_path):
                try:
                    arcpy.Delete_management(output_path)
                except:
                    logging.info('  Error deleting existing albedo raster')
                    continue
            if not arcpy.Exists(output_path):
                logging.debug('  Computing Albedo')
                try:
                    # wb_coef = [0.254, 0.149, 0.147, 0.311, 0.103, 0.036]
                    output_obj = (
                        blue * 0.254 + green * 0.149 + red * 0.147 +
                        nir * 0.311 + swir1 * 0.103 + swir2 * 0.036)
                    arcpy.CopyRaster_management(
                        output_obj, output_path, '#', '#',
                        nodata_value, '#', '#', '32_BIT_FLOAT')
                    # logging.debug('  {}'.format(output_obj))
                    # del output_obj
                except:
                    logging.info('  Error computing Albedo')
                    arcpy.Delete_management(output_path)

            # NDWI (Green / NIR)
            output_path = os.path.join(
                year_ws, refl_sur_path.replace('.refl_sur.tif', '.ndwi_green_nir_sur.tif'))
            if overwrite_flag and os.path.isfile(output_path):
                try:
                    arcpy.Delete_management(output_path)
                except:
                    logging.info('  Error deleting existing NDWI raster')
                    continue
            if not arcpy.Exists(output_path):
                logging.debug('  NDWI (Green / NIR)')
                try:
                    output_obj = (green - nir) / (green + nir)
                    arcpy.CopyRaster_management(
                        output_obj, output_path, '#', '#',
                        nodata_value, '#', '#', '32_BIT_FLOAT')
                    # logging.debug('  {}'.format(output_obj))
                    # del output_obj
                except:
                    logging.info('  Error computing NDWI')
                    arcpy.Delete_management(output_path)

            # NDWI (Green / SWIR1)
            output_path = os.path.join(
                year_ws, refl_sur_path.replace('.refl_sur.tif', '.ndwi_green_swir1_sur.tif'))
            if overwrite_flag and os.path.isfile(output_path):
                try:
                    arcpy.Delete_management(output_path)
                except:
                    logging.info('  Error deleting existing NDWI raster')
                    continue
            if not arcpy.Exists(output_path):
                logging.debug('  NDWI (Green / SWIR1)')
                try:
                    output_obj = (green - swir1) / (green + swir1)
                    arcpy.CopyRaster_management(
                        output_obj, output_path, '#', '#',
                        nodata_value, '#', '#', '32_BIT_FLOAT')
                    # logging.debug('  {}'.format(output_obj))
                    # del output_obj
                except:
                    logging.info('  Error computing NDWI')
                    arcpy.Delete_management(output_path)

            # NDWI (NIR / SWIR1)
            output_path = os.path.join(
                year_ws, refl_sur_path.replace('.refl_sur.tif', '.ndwi_nir_swir1_sur.tif'))
            if overwrite_flag and os.path.isfile(output_path):
                try:
                    arcpy.Delete_management(output_path)
                except:
                    logging.info('  Error deleting existing NDWI raster')
                    continue
            if not arcpy.Exists(output_path):
                logging.debug('  NDWI (NIR / SWIR1)')
                try:
                    output_obj = (nir - swir1) / (nir + swir1)
                    arcpy.CopyRaster_management(
                        output_obj, output_path, '#', '#',
                        nodata_value, '#', '#', '32_BIT_FLOAT')
                    # logging.debug('  {}'.format(output_obj))
                    # del output_obj
                except:
                    logging.info('  Error computing NDWI')
                    arcpy.Delete_management(output_path)

            # # NDWI (SWIR1 / Green)
            # output_path = os.path.join(
            #     year_ws, refl_sur_path.replace('.refl_sur.tif', '.ndwi_swir1_green_sur.tif'))
            # if overwrite_flag and os.path.isfile(output_path):
            #     try:
            #         arcpy.Delete_management(output_path)
            #     except:
            #         logging.info('  Error deleting existing NDWI raster')
            #         continue
            # if not arcpy.Exists(output_path):
            #     logging.debug('  NDWI (SWIR1 / Green)')
            #     try:
            #         output_obj = (swir1 - green) / (swir1 + green)
            #         arcpy.CopyRaster_management(
            #             output_obj, output_path, '#', '#',
            #             nodata_value, '#', '#', '32_BIT_FLOAT')
            #         # logging.debug('  {}'.format(output_obj))
            #         # del output_obj
            #     except:
            #         logging.info('  Error computing NDWI')
            #         arcpy.Delete_management(output_path)

            try:
                del blue, green, red, nir, swir1, swir2
            except:
                logging.info('  Error clearing variables')

            # break

        for file_name in os.listdir(year_ws):
            if file_name.endswith('.tif.xml') or file_name.endswith('.tfw'):
                try:
                    os.remove(os.path.join(year_ws, file_name))
                except:
                    logging.info('  Could not remove {}'.format(file_name))

    try:
        shutil.rmtree(temp_ws)
    except:
        pass


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate Landsat Images from Composite',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()

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

    main(overwrite_flag=args.overwrite)
