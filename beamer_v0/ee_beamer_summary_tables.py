#--------------------------------
# Name:         ee_beamer_summary_tables.py
# Purpose:      Generate Beamer ETg summary figures
# Created       2017-07-27
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
from pandas import ExcelWriter

# This is an awful way of getting the parent folder into the path
# We really should package this up as a module with a setup.py
# This way the ee_tools folders would be in the
#   PYTHONPATH env. variable
ee_tools_path = os.path.dirname(os.path.dirname(
    os.path.abspath(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(ee_tools_path, 'ee_tools'))
sys.path.insert(0, ee_tools_path)
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils


def main(ini_path, overwrite_flag=True):
    """Generate Beamer ETg summary tables

    Args:
        ini_path (str):
        overwrite_flag (bool): if True, overwrite existing figures
            Default is True (for now)
    """

    logging.info('\nGenerate Beamer ETg summary tables')

    # # Eventually get from INI (like ini['BEAMER']['landsat_products'])
    # daily_fields = [
    #     'ZONE_NAME', 'ZONE_FID', 'DATE', 'SCENE_ID', 'PLATFORM', 'PATH', 'ROW',
    #     'YEAR', 'MONTH', 'DAY', 'DOY', 'WATER_YEAR',
    #     'PIXEL_COUNT', 'ETSTAR_COUNT',
    #     'NDVI_TOA', 'NDWI_TOA', 'ALBEDO_SUR', 'TS', 'EVI_SUR', 'ETSTAR_MEAN',
    #     'ETG_MEAN', 'ETG_LPI', 'ETG_UPI', 'ETG_LCI', 'ETG_UCI',
    #     'ET_MEAN', 'ET_LPI', 'ET_UPI', 'ET_LCI', 'ET_UCI',
    #     'WY_ETO', 'WY_PPT']
    # annual_fields = [
    #     'SCENE_COUNT', 'PIXEL_COUNT', 'ETSTAR_COUNT',
    #     'NDVI_TOA', 'NDWI_TOA', 'ALBEDO_SUR', 'TS',
    #     'EVI_SUR_MEAN', 'EVI_SUR_MEDIAN', 'EVI_SUR_MIN', 'EVI_SUR_MAX',
    #     'ETSTAR_MEAN',
    #     'ETG_MEAN', 'ETG_LPI', 'ETG_UPI', 'ETG_LCI', 'ETG_UCI',
    #     'ET_MEAN', 'ET_LPI', 'ET_UPI', 'ET_LCI', 'ET_UCI',
    #     'WY_ETO', 'WY_PPT']

    # For unit conversion
    eto_fields = [
        'ETG_MEAN', 'ETG_LPI', 'ETG_UPI', 'ETG_LCI', 'ETG_UCI',
        'ET_MEAN', 'ET_LPI', 'ET_UPI', 'ET_LCI', 'ET_UCI',
        'WY_ETO']
    ppt_fields = ['WY_PPT']

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='ZONAL_STATS')
    inputs.parse_section(ini, section='BEAMER')
    inputs.parse_section(ini, section='SUMMARY')
    inputs.parse_section(ini, section='TABLES')

    # Output paths
    output_daily_path = os.path.join(
        ini['SUMMARY']['output_ws'],
        ini['BEAMER']['output_name'].replace('.csv', '_daily.xlsx'))
    output_annual_path = os.path.join(
        ini['SUMMARY']['output_ws'],
        ini['BEAMER']['output_name'].replace('.csv', '_annual.xlsx'))

    # Check if files already exist
    if overwrite_flag:
        if os.path.isfile(output_daily_path):
            os.remove(output_daily_path)
        if os.path.isfile(output_annual_path):
            os.remove(output_annual_path)
    else:
        if (os.path.isfile(output_daily_path) and
                os.path.isfile(output_annual_path)):
            logging.info('\nOutput files already exist and '
                         'overwrite is False, exiting')
            return True

    # Start/end year
    year_list = list(range(
        ini['INPUTS']['start_year'], ini['INPUTS']['end_year'] + 1))
    month_list = list(utils.wrapped_range(
        ini['INPUTS']['start_month'], ini['INPUTS']['end_month'], 1, 12))
    doy_list = list(utils.wrapped_range(
        ini['INPUTS']['start_doy'], ini['INPUTS']['end_doy'], 1, 366))

    # GRIDMET month range (default to water year)
    gridmet_start_month = ini['SUMMARY']['gridmet_start_month']
    gridmet_end_month = ini['SUMMARY']['gridmet_end_month']
    gridmet_months = list(utils.month_range(
        gridmet_start_month, gridmet_end_month))
    logging.info('\nGridmet months: {}'.format(
        ', '.join(map(str, gridmet_months))))

    # Read in the zonal stats CSV
    logging.debug('  Reading zonal stats CSV file')
    input_df = pd.read_csv(os.path.join(
        ini['ZONAL_STATS']['output_ws'], ini['BEAMER']['output_name']))

    logging.debug('  Filtering Landsat dataframe')
    input_df = input_df[input_df['PIXEL_COUNT'] > 0]

    # # This assumes that there are L5/L8 images in the dataframe
    # if not input_df.empty:
    #     max_pixel_count = max(input_df['PIXEL_COUNT'])
    # else:
    #     max_pixel_count = 0

    if ini['INPUTS']['fid_keep_list']:
        input_df = input_df[input_df['ZONE_FID'].isin(
            ini['INPUTS']['fid_keep_list'])]
    if ini['INPUTS']['fid_skip_list']:
        input_df = input_df[~input_df['ZONE_FID'].isin(
            ini['INPUTS']['fid_skip_list'])]
    zone_name_list = sorted(list(set(input_df['ZONE_NAME'].values)))

    if year_list:
        input_df = input_df[input_df['YEAR'].isin(year_list)]
    if month_list:
        input_df = input_df[input_df['MONTH'].isin(month_list)]
    if doy_list:
        input_df = input_df[input_df['DOY'].isin(doy_list)]

    if ini['INPUTS']['path_keep_list']:
        input_df = input_df[
            input_df['PATH'].isin(ini['INPUTS']['path_keep_list'])]
    if (ini['INPUTS']['row_keep_list'] and
            ini['INPUTS']['row_keep_list'] != ['XXX']):
        input_df = input_df[
            input_df['ROW'].isin(ini['INPUTS']['row_keep_list'])]

    # Assume the default is for these to be True and only filter if False
    if not ini['INPUTS']['landsat4_flag']:
        input_df = input_df[input_df['PLATFORM'] != 'LT04']
    if not ini['INPUTS']['landsat5_flag']:
        input_df = input_df[input_df['PLATFORM'] != 'LT05']
    if not ini['INPUTS']['landsat7_flag']:
        input_df = input_df[input_df['PLATFORM'] != 'LE07']
    if not ini['INPUTS']['landsat8_flag']:
        input_df = input_df[input_df['PLATFORM'] != 'LC08']

    if ini['INPUTS']['scene_id_keep_list']:
        # Replace XXX with primary ROW value for checking skip list SCENE_ID
        scene_id_df = pd.Series([
            s.replace('XXX', '{:03d}'.format(int(r)))
            for s, r in zip(input_df['SCENE_ID'], input_df['ROW'])])
        input_df = input_df[scene_id_df.isin(
            ini['INPUTS']['scene_id_keep_list']).values]
        # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
        # input_df = input_df[input_df['SCENE_ID'].isin(
        #     ini['INPUTS']['scene_id_keep_list'])]
    if ini['INPUTS']['scene_id_skip_list']:
        # Replace XXX with primary ROW value for checking skip list SCENE_ID
        scene_id_df = pd.Series([
            s.replace('XXX', '{:03d}'.format(int(r)))
            for s, r in zip(input_df['SCENE_ID'], input_df['ROW'])])
        input_df = input_df[np.logical_not(scene_id_df.isin(
            ini['INPUTS']['scene_id_skip_list']).values)]
        # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
        # input_df = input_df[~input_df['SCENE_ID'].isin(
        #     ini['INPUTS']['scene_id_skip_list'])]

    # Filter by QA/QC value
    if ini['SUMMARY']['max_qa'] >= 0 and not input_df.empty:
        logging.debug('    Maximum QA: {0}'.format(
            ini['SUMMARY']['max_qa']))
        input_df = input_df[input_df['QA'] <= ini['SUMMARY']['max_qa']]

    # First filter by average cloud score
    if ini['SUMMARY']['max_cloud_score'] < 100 and not input_df.empty:
        logging.debug('    Maximum cloud score: {0}'.format(
            ini['SUMMARY']['max_cloud_score']))
        input_df = input_df[
            input_df['CLOUD_SCORE'] <= ini['SUMMARY']['max_cloud_score']]

    # Filter by Fmask percentage
    if ini['SUMMARY']['max_fmask_pct'] < 100 and not input_df.empty:
        input_df['FMASK_PCT'] = 100 * (
            input_df['FMASK_COUNT'] / input_df['FMASK_TOTAL'])
        logging.debug('    Max Fmask threshold: {}'.format(
            ini['SUMMARY']['max_fmask_pct']))
        input_df = input_df[
            input_df['FMASK_PCT'] <= ini['SUMMARY']['max_fmask_pct']]

    # Filter low count SLC-off images
    if ini['SUMMARY']['min_slc_off_pct'] > 0 and not input_df.empty:
        logging.debug('    Mininum SLC-off threshold: {}%'.format(
            ini['SUMMARY']['min_slc_off_pct']))
        # logging.debug('    Maximum pixel count: {}'.format(
        #     max_pixel_count))
        slc_off_mask = (
            (input_df['PLATFORM'] == 'LE07') &
            ((input_df['YEAR'] >= 2004) |
             ((input_df['YEAR'] == 2003) & (input_df['DOY'] > 151))))
        slc_off_pct = 100 * (input_df['PIXEL_COUNT'] / input_df['PIXEL_TOTAL'])
        # slc_off_pct = 100 * (input_df['PIXEL_COUNT'] / max_pixel_count)
        input_df = input_df[
            ((slc_off_pct >= ini['SUMMARY']['min_slc_off_pct']) & slc_off_mask) |
            (~slc_off_mask)]

    if input_df.empty:
        logging.error('  Empty dataframe after filtering, exiting')
        return False

    if not os.path.isfile(output_daily_path):
        logging.info('\nWriting daily values to Excel')
        excel_f = ExcelWriter(output_daily_path)
        for zone_name in sorted(zone_name_list):
            logging.info('  {}'.format(zone_name))
            zone_df = input_df[input_df['ZONE_NAME'] == zone_name]
            zone_df.to_excel(
                excel_f, zone_name, index=False, float_format='%.4f')
            # zone_df.to_excel(excel_f, zone_name, index=False)
            del zone_df
        excel_f.save()

    if not os.path.isfile(output_annual_path):
        logging.info('\nComputing annual summaries')
        annual_df = input_df \
            .groupby(['ZONE_NAME', 'YEAR']) \
            .agg({
                'PIXEL_COUNT': ['count', 'mean'],
                'PIXEL_TOTAL': ['mean'],
                'FMASK_COUNT': 'mean',
                'FMASK_TOTAL': 'mean',
                'CLOUD_SCORE': 'mean',
                'ETSTAR_COUNT': 'mean',
                'NDVI_TOA': 'mean',
                'NDWI_TOA': 'mean',
                'ALBEDO_SUR': 'mean',
                'TS': 'mean',
                # 'EVI_SUR': 'mean',
                'EVI_SUR': ['mean', 'median', 'min', 'max'],
                'ETSTAR_MEAN': 'mean',
                'ETG_MEAN': 'mean',
                'ETG_LPI': 'mean',
                'ETG_UPI': 'mean',
                'ETG_LCI': 'mean',
                'ETG_UCI': 'mean',
                'ET_MEAN': 'mean',
                'ET_LPI': 'mean',
                'ET_UPI': 'mean',
                'ET_LCI': 'mean',
                'ET_UCI': 'mean',
                'WY_ETO': 'mean',
                'WY_PPT': 'mean'
            })
        annual_df.columns = annual_df.columns.map('_'.join)
        annual_df = annual_df.rename(columns={
            'PIXEL_COUNT_count': 'SCENE_COUNT',
            'PIXEL_COUNT_mean': 'PIXEL_COUNT'})
        annual_df = annual_df.rename(columns={
            'EVI_SUR_mean': 'EVI_SUR_MEAN',
            'EVI_SUR_median': 'EVI_SUR_MEDIAN',
            'EVI_SUR_min': 'EVI_SUR_MIN',
            'EVI_SUR_max': 'EVI_SUR_MAX'})
        annual_df.rename(
            columns=lambda x: str(x).replace('_mean', ''), inplace=True)
        annual_df['SCENE_COUNT'] = annual_df['SCENE_COUNT'].astype(np.int)
        annual_df['PIXEL_COUNT'] = annual_df['PIXEL_COUNT'].astype(np.int)
        annual_df['PIXEL_TOTAL'] = annual_df['PIXEL_TOTAL'].astype(np.int)
        annual_df['FMASK_COUNT'] = annual_df['FMASK_COUNT'].astype(np.int)
        annual_df['FMASK_TOTAL'] = annual_df['FMASK_TOTAL'].astype(np.int)
        annual_df['ETSTAR_COUNT'] = annual_df['ETSTAR_COUNT'].astype(np.int)
        annual_df = annual_df.reset_index()

        # Convert ETo units
        if (ini['BEAMER']['eto_units'] == 'mm' and
                ini['TABLES']['eto_units'] == 'mm'):
            pass
        elif (ini['BEAMER']['eto_units'] == 'mm' and
                ini['TABLES']['eto_units'] == 'in'):
            annual_df[eto_fields] /= (25.4)
        elif (ini['BEAMER']['eto_units'] == 'mm' and
                ini['TABLES']['eto_units'] == 'ft'):
            annual_df[eto_fields] /= (12 * 25.4)
        else:
            logging.error(
                ('\nERROR: Input units {} and output units {} are not ' +
                 'currently supported, exiting').format(
                    ini['BEAMER']['eto_units'], ini['TABLES']['eto_units']))
            sys.exit()

        # Convert PPT units
        if (ini['BEAMER']['ppt_units'] == 'mm' and
                ini['TABLES']['ppt_units'] == 'mm'):
            pass
        elif (ini['BEAMER']['ppt_units'] == 'mm' and
                ini['TABLES']['ppt_units'] == 'in'):
            annual_df[ppt_fields] /= (25.4)
        elif (ini['BEAMER']['ppt_units'] == 'mm' and
                ini['TABLES']['ppt_units'] == 'ft'):
            annual_df[ppt_fields] /= (12 * 25.4)
        else:
            logging.error(
                ('\nERROR: Input units {} and output units {} are not ' +
                 'currently supported, exiting').format(
                    ini['BEAMER']['ppt_units'], ini['TABLES']['ppt_units']))
            sys.exit()

        logging.info('\nWriting annual values to Excel')
        excel_f = ExcelWriter(output_annual_path)
        for zone_name in sorted(zone_name_list):
            logging.info('  {}'.format(zone_name))
            zone_df = annual_df[annual_df['ZONE_NAME'] == zone_name]
            zone_df.to_excel(
                excel_f, zone_name, index=False, float_format='%.4f')
            del zone_df
        excel_f.save()


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate Beamer ETg summary tables',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    # parser.add_argument(
    #     '-o', '--overwrite', default=False, action='store_true',
    #     help='Force overwrite of existing files')
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

    main(ini_path=args.ini)
