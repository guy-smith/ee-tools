import argparse
import datetime
import logging
import os
import re
import subprocess
import sys

import arcpy
import numpy as np


def main():
    # Hardcoded for now, eventually read this from the INI
    workspace = r'Y:\justinh\Projects\SNWA\ee\images\spring_valley\landsat'

    # year_list = range(1985, 2017)
    year_list = [1984]

    # Float32/Float64
    float_output_type = 'Float32'
    float_nodata_value = np.finfo(np.float32).min
    # int_output_type = 'Byte'
    # int_nodata_value = 255

    raster_re = re.compile('^\d{8}_\d{3}_\w{3}.(tasseled_cap|ts).tif$')

    for year_str in os.listdir(workspace):
        if not re.match('^\d{4}$', year_str):
            continue
        elif int(year_str) not in year_list:
            continue
        logging.info(year_str)

        year_ws = os.path.join(workspace, year_str)
        for file_name in os.listdir(year_ws):
            file_path = os.path.join(year_ws, file_name)
            temp_path = os.path.join(
                year_ws, file_name.replace('.tif', '.temp.tif'))
            if not raster_re.match(file_name):
                continue
            elif arcpy.sa.Raster(file_path).pixelType == 'F64':
                logging.info(file_name)
                # logging.debug(arcpy.sa.Raster(file_path).pixelType)

                # Make a 32 bit float copy
                subprocess.call([
                    'gdalwarp',
                    '-ot', float_output_type, '-overwrite',
                    '-of', 'GTiff', '-co', 'COMPRESS=LZW',
                    # '-srcnodata', str(nodata_value),
                    '-dstnodata', '{:f}'.format(float_nodata_value),
                    file_path, temp_path])
                # Remove the old one
                subprocess.call(['gdalmanage', 'delete', file_path])
                # Rename the file back to the original name
                subprocess.call(['gdalmanage', 'rename', temp_path, file_path])
                # Compute raster statistics
                arcpy.CalculateStatistics_management(file_path)
                os.remove(file_path.replace('.tif', '.tif.xml'))

            # elif arcpy.sa.Raster(file_path).pixelType in ['I16', 'U16']:
            #     print(file_name)
            #     # print(arcpy.sa.Raster(file_path).pixelType)

            #     # Make a 8 bit unsinged integer copy
            #     subprocess.call([
            #         'gdalwarp',
            #         '-ot', int_output_type, '-overwrite',
            #         '-of', 'GTiff', '-co', 'COMPRESS=LZW',
            #         # '-srcnodata', str(nodata_value),
            #         '-dstnodata', int_nodata_value,
            #         file_path, temp_path])
            #     # Remove the old one
            #     subprocess.call(['gdalmanage', 'delete', file_path])
            #     # Rename the file back to the original name
            #     subprocess.call(['gdalmanage', 'rename', temp_path, file_path])
            # break
        # break


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Convert 64-bit Landsat Images to 32-bit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
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

    main()
