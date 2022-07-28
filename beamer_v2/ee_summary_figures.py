#--------------------------------
# Name:         ee_summary_figures.py
# Purpose:      Generate summary figures
# Created       2017-07-13
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
import logging
import os
import sys

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy import stats

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


def main(ini_path=None, overwrite_flag=True, show_flag=False):
    """Generate summary figures

    Args:
        ini_path (str): file path of the control file
        overwrite_flag (bool): if True, overwrite existing figures
        show_flag (bool): if True, show figures as they are being built
    """

    logging.info('\nGenerate summary figures')

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='ZONAL_STATS')
    inputs.parse_section(ini, section='SUMMARY')
    inputs.parse_section(ini, section='FIGURES')

    # Band options
    band_list = [
        'albedo_sur', 'cloud_score', 'eto', 'evi_sur',
        'fmask_count', 'fmask_total', 'ndvi_sur', 'ndvi_toa',
        'ndwi_green_nir_sur', 'ndwi_green_nir_toa',
        'ndwi_green_swir1_sur', 'ndwi_green_swir1_toa',
        'ndwi_nir_swir1_sur', 'ndwi_nir_swir1_toa',
        'ndwi_swir1_green_sur', 'ndwi_swir1_green_toa',
        # 'ndwi_sur', 'ndwi_toa',
        'pixel_count', 'pixel_total', 'ppt',
        'tc_bright', 'tc_green', 'tc_wet', 'ts']
    band_name = {
        'albedo_sur': 'Albedo',
        'cloud_score': 'Cloud Score',
        'eto': 'ETo',
        'evi_sur': 'EVI',
        'fmask_count': 'Fmask Count',
        'fmask_total': 'Fmask Total',
        'ndvi_sur': 'NDVI',
        'ndvi_toa': 'NDVI (TOA)',
        'ndwi_green_nir_sur': 'NDWI (Green, NIR)',
        'ndwi_green_nir_toa': 'NDWI (Green, NIR) (TOA)',
        'ndwi_green_swir1_sur': 'NDWI (Green, SWIR1)',
        'ndwi_green_swir1_toa': 'NDWI (Green, SWIR1) (TOA)',
        'ndwi_nir_swir1_sur': 'NDWI (NIR, SWIR1)',
        'ndwi_nir_swir1_toa': 'NDWI (NIR, SWIR1) (TOA)',
        'ndwi_swir1_green_sur': 'NDWI (SWIR1, Green)',
        'ndwi_swir1_green_toa': 'NDWI (SWIR1, Green) (TOA)',
        # 'ndwi_sur': 'NDWI (SWIR1, GREEN)',
        # 'ndwi_toa': 'NDWI (SWIR1, GREEN) (TOA)',
        'pixel_count': 'Pixel Count',
        'pixel_total': 'Pixel Total',
        'ppt': 'PPT',
        'tc_bright': 'Brightness',
        'tc_green': 'Greeness',
        'tc_wet': 'Wetness',
        'ts': 'Ts'
    }
    band_unit = {
        'albedo_sur': 'dimensionless',
        'cloud_score': 'dimensionless',
        'evi_sur': 'dimensionless',
        'eto': 'mm',
        'fmask_count': 'dimensionless',
        'fmask_total': 'dimensionless',
        'ndvi_sur': 'dimensionless',
        'ndvi_toa': 'dimensionless',
        'ndwi_green_nir_sur': 'dimensionless',
        'ndwi_green_nir_toa': 'dimensionless',
        'ndwi_green_swir1_sur': 'dimensionless',
        'ndwi_green_swir1_toa': 'dimensionless',
        'ndwi_nir_swir1_sur': 'dimensionless',
        'ndwi_nir_swir1_toa': 'dimensionless',
        'ndwi_swir1_green_sur': 'dimensionless',
        'ndwi_swir1_green_toa': 'dimensionless',
        # 'ndwi_sur': 'dimensionless',
        # 'ndwi_toa': 'dimensionless',
        'pixel_count': 'dimensionless',
        'pixel_total': 'dimensionless',
        'ppt': 'mm',
        'tc_bright': 'dimensionless',
        'tc_green': 'dimensionless',
        'tc_wet': 'dimensionless',
        'ts': 'K',
    }
    band_color = {
        'albedo_sur': '#CF4457',
        'cloud_score': '0.5',
        'eto': '#348ABD',
        'fmask_count': '0.5',
        'fmask_total': '0.5',
        'evi_sur': '#FFA500',
        'ndvi_sur': '#A60628',
        'ndvi_toa': '#A60628',
        'ndwi_green_nir_sur': '#4eae4b',
        'ndwi_green_nir_toa': '#4eae4b',
        'ndwi_green_swir1_sur': '#4eae4b',
        'ndwi_green_swir1_toa': '#4eae4b',
        'ndwi_nir_swir1_sur': '#4eae4b',
        'ndwi_nir_swir1_toa': '#4eae4b',
        'ndwi_swir1_green_sur': '#4eae4b',
        'ndwi_swir1_green_toa': '#4eae4b',
        # 'ndwi_sur': '#4eae4b',
        # 'ndwi_toa': '#4eae4b',
        'pixel_count': '0.5',
        'pixel_total': '0.5',
        'ppt': '0.5',
        'tc_bright': '#E24A33',
        'tc_green': '#E24A33',
        'tc_wet': '#E24A33',
        'ts': '#188487'
    }

    # A couple of color palettes to sample from
    # import seaborn as sns
    # print(sns.color_palette('hls', 20).as_hex())
    # print(sns.color_palette('husl', 20).as_hex())
    # print(sns.color_palette('hsv', 20).as_hex())
    # print(sns.color_palette('Set1', 20).as_hex())
    # print(sns.color_palette('Set2', 20).as_hex())

    # Hardcoded plot options
    figures_folder = 'figures'
    fig_type = 'large'

    plot_dict = dict()

    # Center y-labels in figure window (instead of centering on ticks/axes)
    plot_dict['center_ylabel'] = False

    # Axes percentages must be 0-1
    plot_dict['timeseries_band_ax_pct'] = [0.3, 0.92]
    plot_dict['timeseries_ppt_ax_pct'] = [0.0, 0.35]
    plot_dict['complement_band_ax_pct'] = [0.0, 0.5]
    plot_dict['complement_eto_ax_pct'] = [0.4, 1.0]

    if fig_type.lower() == 'large':
        plot_dict['title_fs'] = 12
        plot_dict['xtick_fs'] = 10
        plot_dict['ytick_fs'] = 10
        plot_dict['xlabel_fs'] = 10
        plot_dict['ylabel_fs'] = 10
        plot_dict['legend_fs'] = 10
        plot_dict['ts_ms'] = 3
        plot_dict['comp_ms'] = 4
        plot_dict['timeseries_ax'] = [0.12, 0.13, 0.78, 0.81]
        plot_dict['scatter_ax'] = [0.12, 0.10, 0.82, 0.84]
        plot_dict['complement_ax'] = [0.12, 0.10, 0.78, 0.84]
        plot_dict['fig_size'] = (6.0, 5.0)
    elif fig_type.lower() == 'small':
        plot_dict['title_fs'] = 10
        plot_dict['xtick_fs'] = 8
        plot_dict['ytick_fs'] = 8
        plot_dict['xlabel_fs'] = 8
        plot_dict['ylabel_fs'] = 8
        plot_dict['legend_fs'] = 8
        plot_dict['ts_ms'] = 1.5
        plot_dict['comp_ms'] = 2
        plot_dict['timeseries_ax'] = [0.18, 0.21, 0.67, 0.70]
        plot_dict['scatter_ax'] = [0.18, 0.21, 0.67, 0.70]
        plot_dict['complement_ax'] = [0.18, 0.16, 0.67, 0.75]
        plot_dict['fig_size'] = (3.0, 2.5)
    plot_dict['fig_dpi'] = 300
    plot_dict['show'] = show_flag
    plot_dict['overwrite'] = overwrite_flag


    # CSV parameters
    landsat_annual_fields = [
        'ZONE_FID', 'ZONE_NAME', 'YEAR', 'SCENE_COUNT', 'CLOUD_SCORE',
        'PIXEL_COUNT', 'PIXEL_TOTAL', 'FMASK_COUNT', 'FMASK_TOTAL',
        'TS', 'ALBEDO_SUR', 'NDVI_TOA', 'NDVI_SUR', 'EVI_SUR',
        'NDWI_GREEN_NIR_SUR', 'NDWI_GREEN_SWIR1_SUR', 'NDWI_NIR_SWIR1_SUR',
        # 'NDWI_GREEN_NIR_TOA', 'NDWI_GREEN_SWIR1_TOA', 'NDWI_NIR_SWIR1_TOA',
        # 'NDWI_SWIR1_GREEN_TOA', 'NDWI_SWIR1_GREEN_SUR',
        # 'NDWI_TOA', 'NDWI_SUR',
        'TC_BRIGHT', 'TC_GREEN', 'TC_WET']

    # Add merged row XXX to keep list
    ini['INPUTS']['row_keep_list'].append('XXX')

    # Check figure bands
    timeseries_bands = ini['FIGURES']['timeseries_bands']
    scatter_bands = ini['FIGURES']['scatter_bands']
    complementary_bands = ini['FIGURES']['complementary_bands']
    if timeseries_bands:
        logging.info('Timeseries Bands:')
        for band in timeseries_bands:
            if band not in band_list:
                logging.info(
                    '  Invalid timeseries band: {}, exiting'.format(band))
                return False
            logging.info('  {}'.format(band))
    if scatter_bands:
        logging.info('Scatter Bands (x:y):')
        for band_x, band_y in scatter_bands:
            if band_x not in band_list:
                logging.info(
                    '  Invalid scatter band: {}, exiting'.format(band_x))
                return False
            elif band_y not in band_list:
                logging.info(
                    '  Invalid band: {}, exiting'.format(band_y))
                return False
            logging.info('  {}:{}'.format(band_x, band_y))
    if complementary_bands:
        logging.info('Complementary Bands:')
        for band in complementary_bands:
            if band not in band_list:
                logging.info(
                    '  Invalid complementary band: {}, exiting'.format(band))
                return False
            logging.info('  {}'.format(band))

    # Add input plot options
    plot_dict['ppt_plot_type'] = ini['FIGURES']['ppt_plot_type']
    plot_dict['scatter_best_fit'] = ini['FIGURES']['scatter_best_fit']

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
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone_name = zone_name.replace(' ', '_')
        logging.info('ZONE: {} (FID: {})'.format(zone_name, zone_fid))

        zone_stats_ws = os.path.join(
            ini['ZONAL_STATS']['output_ws'], zone_name)
        zone_figures_ws = os.path.join(
            ini['SUMMARY']['output_ws'], zone_name, figures_folder)
        if not os.path.isdir(zone_stats_ws):
            logging.debug('  Folder {} does not exist, skipping'.format(
                zone_stats_ws))
            continue
        elif not os.path.isdir(zone_figures_ws):
            os.makedirs(zone_figures_ws)

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

        # Output paths
        landsat_summary_path = os.path.join(
            zone_figures_ws, '{}_landsat_figures.csv'.format(zone_name))
        gridmet_summary_path = os.path.join(
            zone_figures_ws, '{}_gridmet_figures.csv'.format(zone_name))
        zone_summary_path = os.path.join(
            zone_figures_ws, '{}_zone_figures.csv'.format(zone_name))

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
            landsat_df = landsat_df[landsat_df['QA'] <= ini['SUMMARY']['max_qa']]

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
                (landsat_df['LANDSAT'] == 'LE7') &
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

        logging.debug('  Computing Landsat annual summaries')
        agg_dict = {
            'PIXEL_COUNT': {
                'PIXEL_COUNT': 'mean',
                'SCENE_COUNT': 'count'},
            'PIXEL_TOTAL': {'PIXEL_TOTAL': 'mean'},
            'FMASK_COUNT': {'FMASK_COUNT': 'mean'},
            'FMASK_TOTAL': {'FMASK_TOTAL': 'mean'},
            'CLOUD_SCORE': {'CLOUD_SCORE': 'mean'}}
        for field in landsat_df.columns.values:
            if field in landsat_annual_fields:
                agg_dict.update({field: {field: 'mean'}})
        landsat_df = landsat_df \
            .groupby(['ZONE_NAME', 'ZONE_FID', 'YEAR']) \
            .agg(agg_dict)
        landsat_df.columns = landsat_df.columns.droplevel(0)
        landsat_df.reset_index(inplace=True)
        # landsat_df = landsat_df[landsat_annual_fields]
        landsat_df['YEAR'] = landsat_df['YEAR'].astype(np.int)
        landsat_df['SCENE_COUNT'] = landsat_df['SCENE_COUNT'].astype(np.int)
        landsat_df['PIXEL_COUNT'] = landsat_df['PIXEL_COUNT'].astype(np.int)
        landsat_df['PIXEL_TOTAL'] = landsat_df['PIXEL_TOTAL'].astype(np.int)
        landsat_df['FMASK_COUNT'] = landsat_df['FMASK_COUNT'].astype(np.int)
        landsat_df['FMASK_TOTAL'] = landsat_df['FMASK_TOTAL'].astype(np.int)
        landsat_df.sort_values(by='YEAR', inplace=True)

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
            .groupby(['ZONE_FID', 'ZONE_NAME', 'GROUP_YEAR']) \
            .agg({'ETO': {'ETO': 'sum'}, 'PPT': {'PPT': 'sum'}})
        gridmet_group_df.columns = gridmet_group_df.columns.droplevel(0)
        gridmet_group_df.reset_index(inplace=True)
        gridmet_group_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
        gridmet_group_df.sort_values(by='YEAR', inplace=True)

        # # Group GRIDMET data by month
        # gridmet_month_df = gridmet_df.groupby(
        #     ['ZONE_FID', 'ZONE_NAME', 'GROUP_YEAR', 'MONTH']).agg({
        #         'ETO': {'ETO': 'sum'}, 'PPT': {'PPT': 'sum'}})
        # gridmet_month_df.columns = gridmet_month_df.columns.droplevel(0)
        # gridmet_month_df.reset_index(inplace=True)
        # gridmet_month_df.rename(columns={'GROUP_YEAR': 'YEAR'}, inplace=True)
        # # gridmet_month_df.sort_values(by=['YEAR', 'MONTH'], inplace=True)
        # gridmet_month_df.reset_index(inplace=True)
        # # Rename monthly PPT columns
        # gridmet_month_df['MONTH'] = 'PPT_M' + gridmet_month_df['MONTH'].astype(str)
        # # Pivot rows up to separate columns
        # gridmet_month_df = gridmet_month_df.pivot_table(
        #     'PPT', ['ZONE_FID', 'YEAR'], 'MONTH')
        # gridmet_month_df.reset_index(inplace=True)
        # columns = ['ZONE_FID', 'YEAR'] + ['PPT_M{}'.format(m) for m in gridmet_months]
        # gridmet_month_df = gridmet_month_df[columns]
        # del gridmet_month_df.index.name


        # Merge Landsat and GRIDMET collections
        zone_df = landsat_df.merge(
            gridmet_group_df, on=['ZONE_FID', 'ZONE_NAME', 'YEAR'])
            # gridmet_group_df, on=['ZONE_FID', 'YEAR'])
        # zone_df = zone_df.merge(
        #     gridmet_month_df, on=['ZONE_FID', 'ZONE_NAME', 'YEAR'])
        #     gridmet_month_df, on=['ZONE_FID', 'YEAR'])
        if zone_df is None or zone_df.empty:
            logging.info('  Empty zone dataframe, not generating figures')
            continue


        # Save annual Landsat and GRIDMET tables
        logging.debug('  Saving summary tables')

        logging.debug('  {}'.format(landsat_summary_path))
        landsat_df.sort_values(by=['YEAR'], inplace=True)
        landsat_df.to_csv(landsat_summary_path, index=False)
        # columns=export_fields

        logging.debug('  {}'.format(gridmet_summary_path))
        gridmet_group_df.sort_values(by=['YEAR'], inplace=True)
        gridmet_group_df.to_csv(gridmet_summary_path, index=False)
        # columns=export_fields

        logging.debug('  {}'.format(zone_summary_path))
        zone_df.sort_values(by=['YEAR'], inplace=True)
        zone_df.to_csv(zone_summary_path, index=False)
        # columns=export_fields


        # Adjust year range based on data availability?
        # start_year = min(zone_df['YEAR']),
        # end_year = max(zone_df['YEAR'])

        logging.debug('  Generating figures')
        for band in timeseries_bands:
            timeseries_plot(
                band, zone_df, zone_name, zone_figures_ws,
                ini['INPUTS']['start_year'], ini['INPUTS']['end_year'],
                band_name, band_unit, band_color, plot_dict)

        for band_x, band_y in scatter_bands:
            scatter_plot(
                band_x, band_y, zone_df, zone_name, zone_figures_ws,
                band_name, band_unit, band_color, plot_dict)

        for band in complementary_bands:
            complementary_plot(
                band, zone_df, zone_name, zone_figures_ws,
                band_name, band_unit, band_color, plot_dict)

        del landsat_df, gridmet_df, zone_df


def timeseries_plot(band, zone_df, zone_name, figures_ws,
                    start_year, end_year,
                    band_name, band_unit, band_color, plot_dict):
    """"""
    ppt_band = 'ppt'
    logging.debug('    Timeseries: {} & {}'.format(
        band_name[band], band_name[ppt_band]))
    figure_path = os.path.join(
        figures_ws,
        '{}_timeseries_{}_&_ppt.png'.format(
            zone_name.lower(), band.lower(), ppt_band))

    fig = plt.figure(figsize=plot_dict['fig_size'])
    fig_ax = plot_dict['timeseries_ax']

    # Position the adjusted axes
    # Draw PPT first so that band lines are on top of PPT bars
    ppt_ax = fig_ax[:]
    ppt_ax[1] = fig_ax[1] + plot_dict['timeseries_ppt_ax_pct'][0] * fig_ax[3]
    ppt_ax[3] = fig_ax[3] * (
        plot_dict['timeseries_ppt_ax_pct'][1] -
        plot_dict['timeseries_ppt_ax_pct'][0])
    ax2 = fig.add_axes(ppt_ax)
    band_ax = fig_ax[:]
    band_ax[1] = fig_ax[1] + plot_dict['timeseries_band_ax_pct'][0] * fig_ax[3]
    band_ax[3] = fig_ax[3] * (
        plot_dict['timeseries_band_ax_pct'][1] -
        plot_dict['timeseries_band_ax_pct'][0])
    ax1 = fig.add_axes(band_ax)

    ax0 = fig.add_axes(fig_ax)
    ax0.set_title('{}'.format(zone_name), size=plot_dict['title_fs'], y=1.01)

    ax2.set_xlabel('Year', fontsize=plot_dict['xlabel_fs'])
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    for tick in ax2.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha('right')
    ax1.set_xlim([start_year - 1, end_year + 1])
    ax2.set_xlim([start_year - 1, end_year + 1])
    ax1.yaxis.set_label_position('left')
    ax2.yaxis.set_label_position('right')
    if plot_dict['center_ylabel']:
        fig.text(
            0.02, 0.5, '{} [{}]'.format(band_name[band], band_unit[band]),
            fontsize=plot_dict['ylabel_fs'],
            ha='center', va='center', rotation='vertical')
        fig.text(
            0.98, 0.5,
            '{} [{}]'.format(band_name[ppt_band], band_unit[ppt_band]),
            fontsize=plot_dict['ylabel_fs'],
            ha='center', va='center', rotation='vertical')
    else:
        ax1.set_ylabel(
            '{} [{}]'.format(band_name[band], band_unit[band]),
            fontsize=plot_dict['ylabel_fs'])
        ax2.set_ylabel(
            '{} [{}]'.format(band_name['ppt'], band_unit['ppt']),
            fontsize=plot_dict['ylabel_fs'])
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax1.tick_params(axis='y', labelsize=plot_dict['ytick_fs'])
    ax2.tick_params(axis='y', labelsize=plot_dict['ytick_fs'])
    ax2.tick_params(axis='x', labelsize=plot_dict['xtick_fs'])
    ax2.tick_params(axis='x', which='both', top='off')
    ax0.axes.get_xaxis().set_ticks([])
    ax0.axes.get_yaxis().set_ticks([])
    ax1.axes.get_xaxis().set_ticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax0.patch.set_visible(False)
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)

    # Plot the band data
    ax1.plot(
        zone_df['YEAR'].values, zone_df[band.upper()].values,
        marker='o', ms=plot_dict['ts_ms'], c=band_color[band.lower()],
        label=band_name[band])

    # Plot precipitation first (so that it is in back)
    if plot_dict['ppt_plot_type'] == 'BAR':
        ax2.bar(
            left=zone_df['YEAR'].values, height=zone_df['PPT'].values,
            align='center', width=1, color='0.6', edgecolor='0.5',
            # left=zone_df['YEAR'].values - 0.5, height=zone_df['PPT'].values,
            # align='edge', width=1, color='0.6', edgecolor='0.5',
            label=band_name[ppt_band])
        # ax2.set_ylim = [min(zone_df['PPT'].values), max(zone_df['PPT'].values)]
    elif plot_dict['ppt_plot_type'] == 'LINE':
        ax2.plot(
            zone_df['YEAR'].values, zone_df['PPT'].values,
            marker='x', c='0.4', ms=plot_dict['ts_ms'], lw=0.7,
            label=band_name[ppt_band])

    # Legend
    h2, l2 = ax2.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    ax0.legend(
        h1 + h2, l1 + l2, loc='upper right', frameon=False,
        fontsize=plot_dict['legend_fs'], numpoints=1)

    if plot_dict['overwrite'] or not os.path.isfile(figure_path):
        fig.savefig(figure_path, dpi=plot_dict['fig_dpi'])
    if plot_dict['show']:
        plt.show()
    plt.close(fig)
    del fig, ax0, ax1, ax2
    return True


def scatter_plot(band_x, band_y, zone_df, zone_name, figures_ws,
                 band_name, band_unit, band_color, plot_dict):
    """"""
    logging.debug('    Scatter: {} vs {}'.format(
        band_name[band_x], band_name[band_y]))
    figure_path = os.path.join(
        figures_ws,
        '{}_scatter_{}_vs_{}.png'.format(
            zone_name.lower(), band_x.lower(), band_y.lower()))

    fig = plt.figure(figsize=plot_dict['fig_size'])
    fig_ax = plot_dict['scatter_ax']
    ax0 = fig.add_axes(fig_ax)
    ax0.set_title('{}'.format(zone_name), size=plot_dict['title_fs'], y=1.01)
    ax0.set_xlabel(
        '{}'.format(band_name[band_x]), fontsize=plot_dict['xlabel_fs'])
    ax0.set_ylabel(
        '{}'.format(band_name[band_y]), fontsize=plot_dict['ylabel_fs'])

    # Regression line
    if plot_dict['scatter_best_fit']:
        m, b, r, p, std_err = stats.linregress(
            zone_df[band_x.upper()].values, zone_df[band_y.upper()].values)
        # m, b = np.polyfit(
        #     zone_df[band_x.upper()].values, zone_df[band_y.upper()].values, 1)
        x = np.array(
            [min(zone_df[band_x.upper()]), max(zone_df[band_x.upper()])])
        ax0.plot(x, m * x + b, '-', c='0.1')
        plt.figtext(0.68, 0.17, ('$y = {0:0.4f}x+{1:0.3f}$'.format(m, b)))
        plt.figtext(0.68, 0.13, ('$R^2\! = {0:0.4f}$'.format(r ** 2)))

    ax0.plot(
        zone_df[band_x.upper()].values, zone_df[band_y.upper()].values,
        linestyle='', marker='o', c='0.5', ms=plot_dict['comp_ms'])

    if plot_dict['overwrite'] or not os.path.isfile(figure_path):
        plt.savefig(figure_path, dpi=plot_dict['fig_dpi'])
    if plot_dict['show']:
        plt.show()
    plt.close(fig)
    del fig, ax0

    return True


def complementary_plot(band, zone_df, zone_name, figures_ws,
                       band_name, band_unit, band_color, plot_dict):
    """"""
    logging.debug('    Complementary: {}'.format(band_name[band]))
    figure_path = os.path.join(
        figures_ws,
        '{}_complementary_{}.png'.format(zone_name.lower(), band.lower()))

    fig = plt.figure(figsize=plot_dict['fig_size'])
    fig_ax = plot_dict['complement_ax']
    ax0 = fig.add_axes(fig_ax)
    ax0.set_title('{}'.format(zone_name), size=plot_dict['title_fs'], y=1.01)

    # Position the adjusted axes
    eto_ax = fig_ax[:]
    eto_ax[1] = fig_ax[1] + plot_dict['complement_eto_ax_pct'][0] * fig_ax[3]
    eto_ax[3] = fig_ax[3] * (
        plot_dict['complement_eto_ax_pct'][1] -
        plot_dict['complement_eto_ax_pct'][0])
    ax1 = fig.add_axes(eto_ax)
    band_ax = fig_ax[:]
    band_ax[1] = fig_ax[1] + plot_dict['complement_band_ax_pct'][0] * fig_ax[3]
    band_ax[3] = fig_ax[3] * (
        plot_dict['complement_band_ax_pct'][1] -
        plot_dict['complement_band_ax_pct'][0])
    ax2 = fig.add_axes(band_ax)

    ax2.set_xlabel(
        '{}'.format(band_name['ppt']), fontsize=plot_dict['xlabel_fs'])
    ax1.yaxis.set_label_position('left')
    ax2.yaxis.set_label_position('right')
    if plot_dict['center_ylabel']:
        fig.text(
            0.02, 0.5, '{} [{}]'.format(band_name['eto'], band_unit['eto']),
            fontsize=plot_dict['ylabel_fs'],
            ha='center', va='center', rotation='vertical')
        fig.text(
            0.98, 0.5, '{} [{}]'.format(band_name[band], band_unit[band]),
            fontsize=plot_dict['ylabel_fs'],
            ha='center', va='center', rotation='vertical')
    else:
        ax1.set_ylabel(
            '{} [{}]'.format(band_name['eto'], band_unit['eto']),
            fontsize=plot_dict['ylabel_fs'])
        ax2.set_ylabel(
            '{} [{}]'.format(band_name[band], band_unit[band]),
            fontsize=plot_dict['ylabel_fs'])
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax1.tick_params(axis='y', labelsize=plot_dict['ytick_fs'])
    ax2.tick_params(axis='y', labelsize=plot_dict['ytick_fs'])
    ax2.tick_params(axis='x', labelsize=plot_dict['xtick_fs'])
    ax2.tick_params(axis='x', which='both', top='off')
    ax0.axes.get_xaxis().set_ticks([])
    ax0.axes.get_yaxis().set_ticks([])
    ax1.axes.get_xaxis().set_ticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.patch.set_visible(False)
    ax2.patch.set_visible(False)

    ax1.set_xlim([min(zone_df['PPT']) - 10, max(zone_df['PPT']) + 10])
    ax2.set_xlim([min(zone_df['PPT']) - 10, max(zone_df['PPT']) + 10])

    ax1.plot(
        zone_df['PPT'].values, zone_df['ETO'].values, label=band_name['eto'],
        linestyle='', marker='^', c=band_color['eto'], ms=plot_dict['comp_ms'])
    ax2.plot(
        zone_df['PPT'].values, zone_df[band.upper()].values,
        linestyle='', marker='o', ms=plot_dict['comp_ms'],
        label=band_name[band], c=band_color[band.lower()])

    # Legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax0.legend(
        h1 + h2, l1 + l2, loc='upper right', frameon=False,
        fontsize=plot_dict['legend_fs'], numpoints=1)

    if plot_dict['overwrite'] or not os.path.isfile(figure_path):
        fig.savefig(figure_path, dpi=plot_dict['fig_dpi'])
    if plot_dict['show']:
        plt.show()
    plt.close(fig)
    del fig, ax0, ax1, ax2
    return True


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate summary figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    parser.add_argument(
        '--show', default=False, action='store_true', help='Show plots')
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
    logging.info('\n{0}'.format('#' * 80))
    log_f = '{0:<20s} {1}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, show_flag=args.show)
    # main(ini_path=args.ini, overwrite_flag=args.overwrite)
