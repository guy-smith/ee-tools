#--------------------------------
# Name:         ee_beamer_summary_figures.py
# Purpose:      Generate Beamer ETg summary figures
# Created       2017-07-27
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
import logging
import math
import os
# import re
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

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


def main(ini_path, show_flag=False, overwrite_flag=True):
    """Generate Beamer ETg summary figures

    Args:
        ini_path (str):
        show_flag (bool): if True, show the figures in the browser.
            Default is False.
        overwrite_flag (bool): if True, overwrite existing figures
            Default is True (for now)
    """

    logging.info('\nGenerate Beamer ETg summary figures')

    # Eventually read from INI
    output_folder = 'figures'
    ncolors = [
        '#348ABD', '#7A68A6', '#A60628', '#467821',
        '#CF4457', '#188487', '#E24A33']

    xtick_fs = 8
    ytick_fs = 8
    xlabel_fs = 8
    ylabel_fs = 8
    ms = 2
    figsize = (3.0, 2.5)

    # For unit conversion
    eto_fields = [
        'ETG_MEAN', 'ETG_LPI', 'ETG_UPI', 'ETG_LCI', 'ETG_UCI',
        'ET_MEAN', 'ET_LPI', 'ET_UPI', 'ET_LCI', 'ET_UCI',
        'ETO']
    ppt_fields = ['PPT']

    # Field names that may be needed for Beamer summary figures
    # Only EVI_SUR is needed to generate old style plots
    # Other variables will be summarized if present
    # ETG and ET will be computed from ETSTAR
    landsat_beamer_fields = [
        'EVI_SUR',
        'NDVI_TOA', 'NDWI_TOA', 'NDVI_SUR', 'ALBEDO_SUR', 'TS',
        'ETSTAR_MEAN', 'ETSTAR_LPI', 'ETSTAR_UPI', 'ETSTAR_LCI', 'ETSTAR_UCI',
        'ETG_MEAN', 'ETG_LPI', 'ETG_UPI', 'ETG_LCI', 'ETG_UCI',
        'ET_MEAN', 'ET_LPI', 'ET_UPI', 'ET_LCI', 'ET_UCI'
    ]

    # Read config file
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    # inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='ZONAL_STATS')
    inputs.parse_section(ini, section='SUMMARY')
    inputs.parse_section(ini, section='FIGURES')
    inputs.parse_section(ini, section='BEAMER')

    # Hardcode GRIDMET month range to the water year
    ini['SUMMARY']['gridmet_start_month'] = 10
    ini['SUMMARY']['gridmet_end_month'] = 9

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
            ini['SUMMARY']['output_ws'], zone_name, output_folder)
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
            zone_figures_ws, '{}_figures.csv'.format(zone_name))

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

        logging.debug('  Computing annual summaries')
        agg_dict = {
            'PIXEL_COUNT': ['count', 'mean'],
            'PIXEL_TOTAL': ['mean'],
            'FMASK_COUNT': 'mean',
            'FMASK_TOTAL': 'mean',
            'CLOUD_SCORE': 'mean',
            'ETSTAR_COUNT': 'mean'}
        for field in landsat_df.columns.values:
            if field in landsat_beamer_fields:
                agg_dict.update({field: 'mean'})
        landsat_df = landsat_df \
            .groupby(['ZONE_NAME', 'ZONE_FID', 'YEAR']) \
            .agg(agg_dict)
        landsat_df.columns = landsat_df.columns.map('_'.join)
        landsat_df = landsat_df.rename(columns={
            'PIXEL_COUNT_count': 'SCENE_COUNT',
            'PIXEL_COUNT_mean': 'PIXEL_COUNT'})
        landsat_df.rename(
            columns=lambda x: str(x).replace('_mean', ''), inplace=True)
        landsat_df['SCENE_COUNT'] = landsat_df['SCENE_COUNT'].astype(np.int)
        landsat_df['PIXEL_COUNT'] = landsat_df['PIXEL_COUNT'].astype(np.int)
        landsat_df['PIXEL_TOTAL'] = landsat_df['PIXEL_TOTAL'].astype(np.int)
        landsat_df['FMASK_COUNT'] = landsat_df['FMASK_COUNT'].astype(np.int)
        landsat_df['FMASK_TOTAL'] = landsat_df['FMASK_TOTAL'].astype(np.int)
        landsat_df['ETSTAR_COUNT'] = landsat_df['ETSTAR_COUNT'].astype(np.int)
        landsat_df = landsat_df.reset_index()

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

        # Save annual Landsat and GRIDMET tables
        logging.debug('  Saving summary tables')
        logging.debug('  {}'.format(zone_summary_path))
        zone_df.sort_values(by=['YEAR'], inplace=True)
        zone_df.to_csv(zone_summary_path, index=False)
        # columns=export_fields

        # logging.debug('  {}'.format(landsat_summary_path))
        # landsat_df.sort_values(by=['YEAR'], inplace=True)
        # landsat_df.to_csv(landsat_summary_path, index=False)
        # # columns=export_fields

        # logging.debug('  {}'.format(gridmet_summary_path))
        # gridmet_group_df.sort_values(by=['YEAR'], inplace=True)
        # gridmet_group_df.to_csv(gridmet_summary_path, index=False)
        # # columns=export_fields

        # ORIGINAL PLOTTING CODE
        # Convert ETo units
        zone_eto_fields = list(set(eto_fields) & set(zone_df.columns.values))
        if zone_eto_fields:
            if (ini['BEAMER']['eto_units'] == 'mm' and
                    ini['FIGURES']['eto_units'] == 'mm'):
                pass
            elif (ini['BEAMER']['eto_units'] == 'mm' and
                    ini['FIGURES']['eto_units'] == 'in'):
                zone_df[zone_eto_fields] /= (25.4)
            elif (ini['BEAMER']['eto_units'] == 'mm' and
                    ini['FIGURES']['eto_units'] == 'ft'):
                zone_df[zone_eto_fields] /= (12 * 25.4)
            else:
                logging.error(
                    ('\nERROR: Input units {} and output units {} are not '
                     'currently supported, exiting').format(
                        ini['BEAMER']['eto_units'],
                        ini['FIGURES']['eto_units']))
                sys.exit()

        # Convert PPT units
        zone_ppt_fields = list(set(ppt_fields) - set(zone_df.columns.values))
        if zone_ppt_fields:
            if (ini['BEAMER']['ppt_units'] == 'mm' and
                    ini['FIGURES']['ppt_units'] == 'mm'):
                pass
            elif (ini['BEAMER']['ppt_units'] == 'mm' and
                    ini['FIGURES']['ppt_units'] == 'in'):
                zone_df[zone_ppt_fields] /= (25.4)
            elif (ini['BEAMER']['ppt_units'] == 'mm' and
                    ini['FIGURES']['ppt_units'] == 'ft'):
                zone_df[zone_ppt_fields] /= (12 * 25.4)
            else:
                logging.error(
                    ('\nERROR: Input units {} and output units {} are not '
                     'currently supported, exiting').format(
                        ini['BEAMER']['ppt_units'], ini['FIGURES']['ppt_units']))
                sys.exit()


        logging.debug('  Generating figures')
        year_min, year_max = min(zone_df['YEAR']), max(zone_df['YEAR'])

        # Set default PPT min/max scaling
        ppt_min = 0
        if ini['FIGURES']['ppt_units'] == 'mm':
            ppt_max = 100 * math.ceil((max(zone_df['PPT']) + 100) / 100)
        elif ini['FIGURES']['ppt_units'] == 'ft':
            ppt_max = 0.2 * math.ceil((max(zone_df['PPT']) + 0.1) / 0.2)
        else:
            ppt_max = 1.2 * max(zone_df['PPT'])


        logging.debug('    EVI vs PPT')
        figure_path = os.path.join(
            zone_figures_ws, '{}_evi.png'.format(zone_name.lower()))
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.19, 0.21, 0.66, 0.75])
        ax1.set_xlabel('Year', fontsize=xlabel_fs)
        ax2 = ax1.twinx()
        ax1.plot(
            zone_df['YEAR'].values, zone_df['PPT'],
            marker='o', c='0.5', ms=ms, label='PPT')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_xlim([year_min - 1, year_max + 1])
        ax1.set_ylim([ppt_min, ppt_max])
        ax1.tick_params(axis='y', labelsize=ytick_fs)
        ax1.tick_params(axis='x', labelsize=xtick_fs)
        ax1.tick_params(axis='x', which='both', top='off')
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        ax1.set_ylabel(
            'PPT [{}/yr]'.format(ini['FIGURES']['ppt_units']),
            fontsize=ylabel_fs)
        ax2.plot(
            zone_df['YEAR'].values, zone_df['EVI_SUR'].values,
            marker='o', c=ncolors[0], ms=ms,
            label='EVI')
        ax1.plot(0, 0, marker='o', c=ncolors[0], ms=ms, label='EVI')
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
        ax2.set_ylim([
            0.05 * math.floor((min(zone_df['EVI_SUR']) - 0.01) / 0.05),
            0.05 * math.ceil((max(zone_df['EVI_SUR']) + 0.01) / 0.05)])
        ax2.tick_params(axis='y', labelsize=ytick_fs)
        ax2.set_ylabel('EVI [dimensionless]', fontsize=ylabel_fs)
        ax1.legend(
            loc='upper right', frameon=False, fontsize=6, numpoints=1)
        if overwrite_flag or not os.path.isfile(figure_path):
            plt.savefig(figure_path, dpi=300)
        plt.close()
        del fig, ax1, ax2


        logging.debug('    ETo vs PPT')
        figure_path = os.path.join(
            zone_figures_ws, '{}_eto.png'.format(zone_name.lower()))
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.18, 0.21, 0.67, 0.75])
        ax1.set_xlabel('Year', fontsize=xlabel_fs)
        ax2 = ax1.twinx()
        ax1.plot(
            zone_df['YEAR'].values, zone_df['PPT'],
            marker='o', c='0.5', ms=ms, label='PPT')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_xlim([year_min - 1, year_max + 1])
        ax1.set_ylim([ppt_min, ppt_max])
        ax1.tick_params(axis='y', labelsize=ytick_fs)
        ax1.tick_params(axis='x', labelsize=xtick_fs)
        ax1.tick_params(axis='x', which='both', top='off')
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        ax1.set_ylabel(
            'PPT [{}/yr]'.format(ini['FIGURES']['ppt_units']),
            fontsize=ylabel_fs)
        ax2.plot(
            zone_df['YEAR'].values, zone_df['ETO'].values,
            marker='o', c=ncolors[1], ms=ms, label='ETo')
        ax1.plot(0, 0, marker='o', c=ncolors[1], ms=ms, label='ETo')
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
        ax2.set_ylim([
            max(0, 0.9 * min(zone_df['ETO'])),
            1.1 * max(zone_df['ETO'])])
        # ax2.set_ylim([
        #     max(0, 100 * math.floor((min(zone_df['ETO']) - 100) / 100)),
        #     100 * math.ceil((max(zone_df['ETO']) + 100) / 100)])
        ax2.tick_params(axis='y', labelsize=ytick_fs)
        ax2.set_ylabel(
            'ETo [{}/yr]'.format(ini['FIGURES']['eto_units']),
            fontsize=ylabel_fs)
        ax1.legend(loc='upper right', frameon=False, fontsize=6, numpoints=1)
        if overwrite_flag or not os.path.isfile(figure_path):
            plt.savefig(figure_path, dpi=300)
        plt.close()
        del fig, ax1, ax2


        logging.debug('    ET vs PPT')
        figure_path = os.path.join(
            zone_figures_ws, '{}_et.png'.format(zone_name.lower()))
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.18, 0.21, 0.67, 0.75])
        ax1.set_xlabel('Year', fontsize=xlabel_fs)
        ax2 = ax1.twinx()
        ax1.plot(
            zone_df['YEAR'].values, zone_df['PPT'],
            marker='o', c='0.5', ms=ms, label='PPT')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_xlim([year_min - 1, year_max + 1])
        ax1.set_ylim([ppt_min, ppt_max])
        ax1.tick_params(axis='y', labelsize=ytick_fs)
        ax1.tick_params(axis='x', labelsize=xtick_fs)
        ax1.tick_params(axis='x', which='both', top='off')
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        ax1.set_ylabel(
            'PPT [{}/yr]'.format(ini['FIGURES']['ppt_units']),
            fontsize=ylabel_fs)
        ax2.plot(
            zone_df['YEAR'].values, zone_df['ET_UCI'].values,
            marker='', c=ncolors[2], alpha=0.5, lw=0.75)
        ax2.plot(
            zone_df['YEAR'].values, zone_df['ET_LCI'].values,
            marker='', c=ncolors[2], alpha=0.5, lw=0.75)
        ax2.plot(
            zone_df['YEAR'].values, zone_df['ET_MEAN'].values,
            marker='o', c=ncolors[2], ms=ms, label='ET')
        ax1.plot(0, 0, marker='o', c=ncolors[2], ms=ms, label='ET')
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
        ax2.set_ylim([
            max(0, 0.9 * min(zone_df['ET_LCI'])),
            1.1 * max(zone_df['ET_UCI'])])
        # ax2.set_ylim([
        #     max(0, 100 * math.floor((min(zone_df['ET_MEAN']) - 100) / 100)),
        #     100 * math.ceil((max(zone_df['ET_MEAN']) + 100) / 100)])
        ax2.tick_params(axis='y', labelsize=ytick_fs)
        ax2.set_ylabel(
            'ET [{}/yr]'.format(ini['FIGURES']['eto_units']),
            fontsize=ylabel_fs)
        ax1.legend(loc='upper right', frameon=False, fontsize=6, numpoints=1)
        if overwrite_flag or not os.path.isfile(figure_path):
            plt.savefig(figure_path, dpi=300)
        plt.close()
        del fig, ax1, ax2


        logging.debug('    ETg vs PPT')
        figure_path = os.path.join(
            zone_figures_ws, '{}_etg.png'.format(zone_name.lower()))
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.18, 0.21, 0.67, 0.75])
        ax1.set_xlabel('Year', fontsize=xlabel_fs)
        ax2 = ax1.twinx()
        ax1.plot(
            zone_df['YEAR'].values, zone_df['PPT'],
            marker='o', c='0.5', ms=ms, label='PPT')
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_xlim([year_min - 1, year_max + 1])
        ax1.set_ylim([ppt_min, ppt_max])
        ax1.tick_params(axis='y', labelsize=ytick_fs)
        ax1.tick_params(axis='x', labelsize=xtick_fs)
        ax1.tick_params(axis='x', which='both', top='off')
        ax1.xaxis.set_minor_locator(MultipleLocator(1))
        for tick in ax1.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
        ax1.set_ylabel(
            'PPT [{}/yr]'.format(ini['FIGURES']['ppt_units']),
            fontsize=ylabel_fs)
        ax2.plot(
            zone_df['YEAR'].values, zone_df['ETG_UCI'].values,
            marker='', c=ncolors[3], alpha=0.5, lw=0.75)
        ax2.plot(
            zone_df['YEAR'].values, zone_df['ETG_LCI'].values,
            marker='', c=ncolors[3], alpha=0.5, lw=0.75)
        ax2.plot(
            zone_df['YEAR'].values, zone_df['ETG_MEAN'].values,
            marker='o', c=ncolors[3], ms=ms, label='ETg')
        ax1.plot(0, 0, marker='o', c=ncolors[3], ms=ms, label='ETg')
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
        ax2.set_ylim([
            max(0, 0.9 * min(zone_df['ETG_LCI'])),
            1.1 * max(zone_df['ETG_UCI'])])
        # ax2.set_ylim([
        #     max(0, 100 * math.floor((min(zone_df['ETG_MEAN']) - 100) / 100)),
        #     100 * math.ceil((max(zone_df['ETG_MEAN']) + 100) / 100)])
        ax2.tick_params(axis='y', labelsize=ytick_fs)
        ax2.set_ylabel(
            'ETg [{}/yr]'.format(ini['FIGURES']['eto_units']),
            fontsize=ylabel_fs)
        ax1.legend(loc='upper right', frameon=False, fontsize=6, numpoints=1)
        if overwrite_flag or not os.path.isfile(figure_path):
            plt.savefig(figure_path, dpi=300)
        plt.close()
        del fig, ax1, ax2


        logging.debug('    Complimentary')
        figure_path = os.path.join(
            zone_figures_ws, '{}_complimentary.png'.format(zone_name.lower()))
        fig = plt.figure(figsize=(3, 2.5))
        ax = fig.add_axes([0.18, 0.16, 0.78, 0.80])
        # ax = fig.add_axes([0.18, 0.21, 0.67, 0.70])
        ax.plot(
            zone_df['PPT'].values, zone_df['ETO'].values,
            linestyle='', marker='o', c=ncolors[1], ms=3, label='ETo')
        ax.plot(
            zone_df['PPT'].values, zone_df['ET_MEAN'].values,
            linestyle='', marker='o', c=ncolors[2], ms=3, label='ET')
        # xmax = 100 * math.ceil(max(zone_df['PPT']) / 100)
        # ymax = 200 * math.ceil((max(zone_df['ETO']) + 200) / 200)
        ax.set_xlim([ppt_min, ppt_max])
        ax.set_ylim([0, 1.2 * max(zone_df['ETO'])])
        ax.tick_params(axis='y', labelsize=ytick_fs)
        ax.tick_params(axis='x', labelsize=xtick_fs)
        ax.tick_params(axis='x', which='both', top='off')
        ax.tick_params(axis='y', which='both', right='off')
        ax.set_xlabel('PPT [{}/yr]'.format(
            ini['FIGURES']['ppt_units']), fontsize=xlabel_fs)
        ax.set_ylabel('ET and ETo [{}/yr]'.format(
            ini['FIGURES']['eto_units']), fontsize=ylabel_fs)
        ax.legend(loc='upper right', frameon=False, fontsize=6, numpoints=1)
        if overwrite_flag or not os.path.isfile(figure_path):
            plt.savefig(figure_path, dpi=300)
        plt.close()
        del fig, ax

        # break


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate Beamer ETg summary figures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '--show', default=False, action='store_true',
        help='Show figures')
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

    main(ini_path=args.ini, show_flag=args.show)
