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
        overwrite_flag (bool): if True, overwrite existing tables
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
    #     'ETO', 'PPT']
    # annual_fields = [
    #     'SCENE_COUNT', 'PIXEL_COUNT', 'ETSTAR_COUNT',
    #     'NDVI_TOA', 'NDWI_TOA', 'ALBEDO_SUR', 'TS',
    #     'EVI_SUR_MEAN', 'EVI_SUR_MEDIAN', 'EVI_SUR_MIN', 'EVI_SUR_MAX',
    #     'ETSTAR_MEAN',
    #     'ETG_MEAN', 'ETG_LPI', 'ETG_UPI', 'ETG_LCI', 'ETG_UCI',
    #     'ET_MEAN', 'ET_LPI', 'ET_UPI', 'ET_LCI', 'ET_UCI',
    #     'ETO', 'PPT']

    # For unit conversion
    eto_fields = [
        'ETG_MEAN', 'ETG_LPI', 'ETG_UPI', 'ETG_LCI', 'ETG_UCI',
        'ET_MEAN', 'ET_LPI', 'ET_UPI', 'ET_LCI', 'ET_UCI',
        'ETO']
    ppt_fields = ['PPT']

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='ZONAL_STATS')
    inputs.parse_section(ini, section='BEAMER')
    inputs.parse_section(ini, section='SUMMARY')
    inputs.parse_section(ini, section='TABLES')

    # Hardcode GRIDMET month range to the water year
    ini['SUMMARY']['gridmet_start_month'] = 10
    ini['SUMMARY']['gridmet_end_month'] = 9

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

    # Get ee features from shapefile
    zone_geom_list = gdc.shapefile_2_geom_list_func(
        ini['INPUTS']['zone_shp_path'], zone_field=ini['INPUTS']['zone_field'],
        reverse_flag=False)

    # Filter features by FID before merging geometries
    if ini['INPUTS']['fid_keep_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] in ini['INPUTS']['fid_keep_list']]
    if ini['INPUTS']['fid_skip_list']:
        zone_geom_list = [
            zone_obj for zone_obj in zone_geom_list
            if zone_obj[0] not in ini['INPUTS']['fid_skip_list']]

    # # Filter features by FID before merging geometries
    # if ini['INPUTS']['fid_keep_list']:
    #     landsat_df = landsat_df[landsat_df['ZONE_FID'].isin(
    #         ini['INPUTS']['fid_keep_list'])]
    # if ini['INPUTS']['fid_skip_list']:
    #     landsat_df = landsat_df[~landsat_df['ZONE_FID'].isin(
    #         ini['INPUTS']['fid_skip_list'])]

    logging.info('\nProcessing zones')
    zone_df_dict = {}
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone_name = zone_name.replace(' ', '_')
        logging.info('ZONE: {} (FID: {})'.format(zone_name, zone_fid))

        zone_stats_ws = os.path.join(
            ini['ZONAL_STATS']['output_ws'], zone_name)
        if not os.path.isdir(zone_stats_ws):
            logging.debug('  Folder {} does not exist, skipping'.format(
                zone_stats_ws))
            continue

        # Input paths
        landsat_daily_path = os.path.join(
            zone_stats_ws, '{}_landsat_daily.csv'.format(zone_name))
        gridmet_daily_path = os.path.join(
            zone_stats_ws, '{}_gridmet_daily.csv'.format(zone_name))
        gridmet_monthly_path = os.path.join(
            zone_stats_ws, '{}_gridmet_monthly.csv'.format(zone_name))
        if not os.path.isfile(landsat_daily_path):
            logging.error('  Landsat daily CSV does not exist, skipping zone')
            continue
        elif (not os.path.isfile(gridmet_daily_path) and
              not os.path.isfile(gridmet_monthly_path)):
            logging.error(
                '  GRIDMET daily or monthly CSV does not exist, skipping zone')
            continue
            # DEADBEEF - Eventually support generating only Landsat figures
            # logging.error(
            #     '  GRIDMET daily and/or monthly CSV files do not exist.\n'
            #     '  ETo and PPT will not be processed.')

        logging.debug('  Reading Landsat CSV')
        landsat_df = pd.read_csv(landsat_daily_path)

        logging.debug('  Filtering Landsat dataframe')
        landsat_df = landsat_df[landsat_df['PIXEL_COUNT'] > 0]

        # QA field should have been written in zonal stats code
        # Eventually this block can be removed
        if 'QA' not in landsat_df.columns.values:
            landsat_df['QA'] = 0

        # # This assumes that there are L5/L8 images in the dataframe
        # if not landsat_df.empty:
        #     max_pixel_count = max(landsat_df['PIXEL_COUNT'])
        # else:
        #     max_pixel_count = 0

        if year_list:
            landsat_df = landsat_df[landsat_df['YEAR'].isin(year_list)]
        if month_list:
            landsat_df = landsat_df[landsat_df['MONTH'].isin(month_list)]
        if doy_list:
            landsat_df = landsat_df[landsat_df['DOY'].isin(doy_list)]

        # Assume the default is for these to be True and only filter if False
        if not ini['INPUTS']['landsat4_flag']:
            landsat_df = landsat_df[landsat_df['PLATFORM'] != 'LT04']
        if not ini['INPUTS']['landsat5_flag']:
            landsat_df = landsat_df[landsat_df['PLATFORM'] != 'LT05']
        if not ini['INPUTS']['landsat7_flag']:
            landsat_df = landsat_df[landsat_df['PLATFORM'] != 'LE07']
        if not ini['INPUTS']['landsat8_flag']:
            landsat_df = landsat_df[landsat_df['PLATFORM'] != 'LC08']

        if ini['INPUTS']['path_keep_list']:
            landsat_df = landsat_df[
                landsat_df['PATH'].isin(ini['INPUTS']['path_keep_list'])]
        if (ini['INPUTS']['row_keep_list'] and
                ini['INPUTS']['row_keep_list'] != ['XXX']):
            landsat_df = landsat_df[
                landsat_df['ROW'].isin(ini['INPUTS']['row_keep_list'])]

        if ini['INPUTS']['scene_id_keep_list']:
            # Replace XXX with primary ROW value for checking skip list SCENE_ID
            scene_id_df = pd.Series([
                s.replace('XXX', '{:03d}'.format(int(r)))
                for s, r in zip(landsat_df['SCENE_ID'], landsat_df['ROW'])])
            landsat_df = landsat_df[scene_id_df.isin(
                ini['INPUTS']['scene_id_keep_list']).values]
            # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
            # landsat_df = landsat_df[landsat_df['SCENE_ID'].isin(
            #     ini['INPUTS']['scene_id_keep_list'])]
        if ini['INPUTS']['scene_id_skip_list']:
            # Replace XXX with primary ROW value for checking skip list SCENE_ID
            scene_id_df = pd.Series([
                s.replace('XXX', '{:03d}'.format(int(r)))
                for s, r in zip(landsat_df['SCENE_ID'], landsat_df['ROW'])])
            landsat_df = landsat_df[np.logical_not(scene_id_df.isin(
                ini['INPUTS']['scene_id_skip_list']).values)]
            # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
            # landsat_df = landsat_df[np.logical_not(landsat_df['SCENE_ID'].isin(
            #     ini['INPUTS']['scene_id_skip_list']))]

        # Filter by QA/QC value
        if ini['SUMMARY']['max_qa'] >= 0 and not landsat_df.empty:
            logging.debug('    Maximum QA: {0}'.format(
                ini['SUMMARY']['max_qa']))
            landsat_df = landsat_df[
                landsat_df['QA'] <= ini['SUMMARY']['max_qa']]

        # Filter by average cloud score
        if ini['SUMMARY']['max_cloud_score'] < 100 and not landsat_df.empty:
            logging.debug('    Maximum cloud score: {0}'.format(
                ini['SUMMARY']['max_cloud_score']))
            landsat_df = landsat_df[
                landsat_df['CLOUD_SCORE'] <= ini['SUMMARY']['max_cloud_score']]

        # Filter by Fmask percentage
        if ini['SUMMARY']['max_fmask_pct'] < 100 and not landsat_df.empty:
            landsat_df['FMASK_PCT'] = 100 * (
                landsat_df['FMASK_COUNT'] / landsat_df['FMASK_TOTAL'])
            logging.debug('    Max Fmask threshold: {}'.format(
                ini['SUMMARY']['max_fmask_pct']))
            landsat_df = landsat_df[
                landsat_df['FMASK_PCT'] <= ini['SUMMARY']['max_fmask_pct']]

        # Filter low count SLC-off images
        if ini['SUMMARY']['min_slc_off_pct'] > 0 and not landsat_df.empty:
            logging.debug('    Mininum SLC-off threshold: {}%'.format(
                ini['SUMMARY']['min_slc_off_pct']))
            # logging.debug('    Maximum pixel count: {}'.format(
            #     max_pixel_count))
            slc_off_mask = (
                (landsat_df['PLATFORM'] == 'LE07') &
                ((landsat_df['YEAR'] >= 2004) |
                 ((landsat_df['YEAR'] == 2003) & (landsat_df['DOY'] > 151))))
            slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / landsat_df['PIXEL_TOTAL'])
            # slc_off_pct = 100 * (landsat_df['PIXEL_COUNT'] / max_pixel_count)
            landsat_df = landsat_df[
                ((slc_off_pct >= ini['SUMMARY']['min_slc_off_pct']) & slc_off_mask) |
                (~slc_off_mask)]

        if landsat_df.empty:
            logging.error(
                '  Empty Landsat dataframe after filtering, skipping zone')
            continue

        # Aggregate GRIDMET (to water year)
        if os.path.isfile(gridmet_monthly_path):
            logging.debug('  Reading montly GRIDMET CSV')
            gridmet_df = pd.read_csv(gridmet_monthly_path)
        elif os.path.isfile(gridmet_daily_path):
            logging.debug('  Reading daily GRIDMET CSV')
            gridmet_df = pd.read_csv(gridmet_daily_path)

        logging.debug('  Computing GRIDMET summaries')
        # Summarize GRIDMET for target months year
        if (gridmet_start_month in [10, 11, 12] and
                gridmet_end_month in [10, 11, 12]):
            month_mask = (
                (gridmet_df['MONTH'] >= gridmet_start_month) &
                (gridmet_df['MONTH'] <= gridmet_end_month))
            gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR'] + 1
        elif (gridmet_start_month in [10, 11, 12] and
              gridmet_end_month not in [10, 11, 12]):
            month_mask = gridmet_df['MONTH'] >= gridmet_start_month
            gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR'] + 1
            month_mask = gridmet_df['MONTH'] <= gridmet_end_month
            gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR']
        else:
            month_mask = (
                (gridmet_df['MONTH'] >= gridmet_start_month) &
                (gridmet_df['MONTH'] <= gridmet_end_month))
            gridmet_df.loc[month_mask, 'GROUP_YEAR'] = gridmet_df['YEAR']
        # GROUP_YEAR for rows not in the GRIDMET month range will be NAN
        gridmet_df = gridmet_df[~pd.isnull(gridmet_df['GROUP_YEAR'])]

        if year_list:
            gridmet_df = gridmet_df[gridmet_df['GROUP_YEAR'].isin(year_list)]

        if gridmet_df.empty:
            logging.error(
                '    Empty GRIDMET dataframe after filtering by year')
            continue

        # Group GRIDMET data by user specified range (default is water year)
        gridmet_group_df = gridmet_df \
            .groupby(['ZONE_NAME', 'ZONE_FID', 'GROUP_YEAR']) \
            .agg({'ETO': np.sum, 'PPT': np.sum}) \
            .reset_index() \
            .sort_values(by='GROUP_YEAR')
            # .rename(columns={'ETO': 'ETO', 'PPT': 'PPT'}) \
        # Rename wasn't working when chained...
        gridmet_group_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
        gridmet_group_df['YEAR'] = gridmet_group_df['YEAR'].astype(int)

        # # Group GRIDMET data by month
        # gridmet_month_df = gridmet_df\
        #     .groupby(['ZONE_NAME', 'ZONE_FID', 'GROUP_YEAR', 'MONTH']) \
        #     .agg({'ETO': np.sum, 'PPT': np.sum}) \
        #     .reset_index() \
        #     .sort_values(by=['GROUP_YEAR', 'MONTH'])
        # gridmet_month_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
        # # Rename monthly PPT columns
        # gridmet_month_df['MONTH'] = 'PPT_M' + gridmet_month_df['MONTH'].astype(str)
        # # Pivot rows up to separate columns
        # gridmet_month_df = gridmet_month_df.pivot_table(
        #     'PPT', ['ZONE_NAME', 'YEAR'], 'MONTH')
        # gridmet_month_df.reset_index(inplace=True)
        # columns = ['ZONE_NAME', 'YEAR'] + ['PPT_M{}'.format(m) for m in gridmet_months]
        # gridmet_month_df = gridmet_month_df[columns]
        # del gridmet_month_df.index.name

        # Merge Landsat and GRIDMET collections
        zone_df = landsat_df.merge(
            gridmet_group_df, on=['ZONE_NAME', 'ZONE_FID', 'YEAR'])
        if zone_df is None or zone_df.empty:
            logging.info('  Empty zone dataframe, not generating figures')
            continue

        # Compute ETg
        zone_df['ETG_MEAN'] = zone_df['ETSTAR_MEAN'] * (
            zone_df['ETO'] - zone_df['PPT'])
        zone_df['ETG_LPI'] = zone_df['ETSTAR_LPI'] * (
            zone_df['ETO'] - zone_df['PPT'])
        zone_df['ETG_UPI'] = zone_df['ETSTAR_UPI'] * (
            zone_df['ETO'] - zone_df['PPT'])
        zone_df['ETG_LCI'] = zone_df['ETSTAR_LCI'] * (
            zone_df['ETO'] - zone_df['PPT'])
        zone_df['ETG_UCI'] = zone_df['ETSTAR_UCI'] * (
            zone_df['ETO'] - zone_df['PPT'])

        # Compute ET
        zone_df['ET_MEAN'] = zone_df['ETG_MEAN'] + zone_df['PPT']
        zone_df['ET_LPI'] = zone_df['ETG_LPI'] + zone_df['PPT']
        zone_df['ET_UPI'] = zone_df['ETG_UPI'] + zone_df['PPT']
        zone_df['ET_LCI'] = zone_df['ETG_LCI'] + zone_df['PPT']
        zone_df['ET_UCI'] = zone_df['ETG_UCI'] + zone_df['PPT']

        # Append zone dataframes
        zone_df_dict[zone_name] = zone_df

    # Export each zone to a separate tab
    if not os.path.isfile(output_daily_path):
        logging.info('\nWriting daily values to Excel')
        excel_f = ExcelWriter(output_daily_path)
        for zone_name, zone_df in sorted(zone_df_dict.items()):
            logging.info('  {}'.format(zone_name))
            zone_df.to_excel(
                excel_f, zone_name, index=False, float_format='%.4f')
            # zone_df.to_excel(excel_f, zone_name, index=False)
            del zone_df
        excel_f.save()

    if not os.path.isfile(output_annual_path):
        logging.info('\nComputing annual summaries')
        annual_df = pd.concat(list(zone_df_dict.values())) \
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
                'ETO': 'mean',
                'PPT': 'mean'
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
        for zone_name in sorted(zone_df_dict.keys()):
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
        'Start Time:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini)
