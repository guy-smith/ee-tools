import argparse
import datetime
import logging
import os
import re
import sys

import arcpy
# import numpy as np


def main(overwrite_flag=False):
    # Hardcoded for now, eventually read this from the INI
    workspace = r'Y:\justinh\Projects\SNWA\ee\images\spring_valley\landsat'

    arcpy.CheckOutExtension('Spatial')
    # arcpy.env.workspace = workspace
    arcpy.env.overwriteOutput = True
    arcpy.env.pyramid = 'PYRAMIDS 0'
    arcpy.env.compression = "LZW"
    arcpy.env.workspace = 'C:\Temp'
    arcpy.env.scratchWorkspace = 'C:\Temp'

    # year_list = range(1985, 2017)
    year_list = [1984]

    for year_str in os.listdir(workspace):
        if not re.match('^\d{4}$', year_str):
            continue
        elif int(year_str) not in year_list:
            continue
        logging.info(year_str)

        year_ws = os.path.join(workspace, year_str)
        for file_name in os.listdir(year_ws):
            file_path = os.path.join(year_ws, file_name)
            if (file_name.endswith('.ts.tif') or
                    file_name.endswith('.tasseled_cap.tif')):
                logging.info(file_name)
                arcpy.CalculateStatistics_management(file_path)
                try:
                    os.remove(file_path.replace('.tif', '.tif.xml'))
                except:
                    pass
            else:
                continue


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
