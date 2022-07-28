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


def main(ini_path, show_flag=False, overwrite_flag=False):
    """Generate Beamer ETg summary figures

    Args:
        ini_path (str):
        show_flag (bool): if True, show the figures in the browser.
            Default is False.
        overwrite_flag (bool): if True, overwrite existing figures
            Default is True (for now)
    """

    logging.info('\nGenerate Beamer ETg summary figures')

    ncolors = [
        '#348ABD', '#7A68A6', '#A60628', '#467821',
        '#CF4457', '#188487', '#E24A33']
    xtick_fs = 8
    ytick_fs = 8
    xlabel_fs = 8
    ylabel_fs = 8
    ms = 2
    figsize = (3.0, 2.5)
    output_folder = 'figures'

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
    inputs.parse_section(ini, section='SUMMARY')
    inputs.parse_section(ini, section='FIGURES')
    inputs.parse_section(ini, section='BEAMER')

    # Output paths
    output_ws = os.path.join(
        ini['SUMMARY']['output_ws'], output_folder)
    if not os.path.isdir(output_ws):
        os.makedirs(output_ws)

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
    logging.debug(input_df.head())

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


    # Process each zone separately
    logging.debug(input_df.head())
    zone_name_list = sorted(list(set(input_df['ZONE_NAME'].values)))
    for zone_name in zone_name_list:
        logging.info('ZONE: {}'.format(zone_name))
        # The names are currently stored in the CSV with spaces
        zone_output_name = zone_name.replace(' ', '_')
        zone_df = input_df[input_df['ZONE_NAME'] == zone_name]
        if zone_df.empty:
            logging.info('  Empty zone dataframe, skipping zone')
            continue


        logging.debug('  Computing annual summaries')
        annual_df = zone_df \
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
                'EVI_SUR': 'mean',
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
                ini['FIGURES']['eto_units'] == 'mm'):
            pass
        elif (ini['BEAMER']['eto_units'] == 'mm' and
                ini['FIGURES']['eto_units'] == 'in'):
            annual_df[eto_fields] /= (25.4)
        elif (ini['BEAMER']['eto_units'] == 'mm' and
                ini['FIGURES']['eto_units'] == 'ft'):
            annual_df[eto_fields] /= (12 * 25.4)
        else:
            logging.error(
                ('\nERROR: Input units {} and output units {} are not ' +
                 'currently supported, exiting').format(
                    ini['BEAMER']['eto_units'], ini['FIGURES']['eto_units']))
            sys.exit()

        # Convert PPT units
        if (ini['BEAMER']['ppt_units'] == 'mm' and
                ini['FIGURES']['ppt_units'] == 'mm'):
            pass
        elif (ini['BEAMER']['ppt_units'] == 'mm' and
                ini['FIGURES']['ppt_units'] == 'in'):
            annual_df[ppt_fields] /= (25.4)
        elif (ini['BEAMER']['ppt_units'] == 'mm' and
                ini['FIGURES']['ppt_units'] == 'ft'):
            annual_df[ppt_fields] /= (12 * 25.4)
        else:
            logging.error(
                ('\nERROR: Input units {} and output units {} are not '
                 'currently supported, exiting').format(
                    ini['BEAMER']['ppt_units'], ini['FIGURES']['ppt_units']))
            sys.exit()


        logging.debug('  Generating figures')
        zone_df = annual_df[annual_df['ZONE_NAME'] == zone_name]
        year_min, year_max = min(zone_df['YEAR']), max(zone_df['YEAR'])

        # Set default PPT min/max scaling
        ppt_min = 0
        if ini['FIGURES']['ppt_units'] == 'mm':
            ppt_max = 100 * math.ceil((max(zone_df['WY_PPT']) + 100) / 100)
        elif ini['FIGURES']['ppt_units'] == 'ft':
            ppt_max = 0.2 * math.ceil((max(zone_df['WY_PPT']) + 0.1) / 0.2)
        else:
            ppt_max = 1.2 * max(zone_df['WY_PPT'])


        logging.debug('    EVI vs PPT')
        figure_path = os.path.join(
            output_ws, '{}_evi.png'.format(zone_output_name))
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.20, 0.21, 0.65, 0.75])
        ax1.set_xlabel('Year', fontsize=xlabel_fs)
        ax2 = ax1.twinx()
        ax1.plot(
            zone_df['YEAR'].values, zone_df['WY_PPT'],
            marker='o', c='0.5', ms=ms, label='WY PPT')
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
            output_ws, '{}_eto.png'.format(zone_output_name))
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.18, 0.21, 0.67, 0.75])
        ax1.set_xlabel('Year', fontsize=xlabel_fs)
        ax2 = ax1.twinx()
        ax1.plot(
            zone_df['YEAR'].values, zone_df['WY_PPT'],
            marker='o', c='0.5', ms=ms, label='WY PPT')
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
            zone_df['YEAR'].values, zone_df['WY_ETO'].values,
            marker='o', c=ncolors[1], ms=ms, label='ETo')
        ax1.plot(0, 0, marker='o', c=ncolors[1], ms=ms, label='ETo')
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position("left")
        ax2.set_ylim([
            max(0, 0.9 * min(zone_df['WY_ETO'])),
            1.1 * max(zone_df['WY_ETO'])])
        # ax2.set_ylim([
        #     max(0, 100 * math.floor((min(zone_df['WY_ETO']) - 100) / 100)),
        #     100 * math.ceil((max(zone_df['WY_ETO']) + 100) / 100)])
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
            output_ws, '{}_et.png'.format(zone_output_name))
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.18, 0.21, 0.67, 0.75])
        ax1.set_xlabel('Year', fontsize=xlabel_fs)
        ax2 = ax1.twinx()
        ax1.plot(
            zone_df['YEAR'].values, zone_df['WY_PPT'],
            marker='o', c='0.5', ms=ms, label='WY PPT')
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
            output_ws, '{}_etg.png'.format(zone_output_name))
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.18, 0.21, 0.67, 0.75])
        ax1.set_xlabel('Year', fontsize=xlabel_fs)
        ax2 = ax1.twinx()
        ax1.plot(
            zone_df['YEAR'].values, zone_df['WY_PPT'],
            marker='o', c='0.5', ms=ms, label='WY PPT')
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
            output_ws, '{}_complimentary.png'.format(zone_output_name))
        fig = plt.figure(figsize=(3, 2.5))
        ax = fig.add_axes([0.18, 0.16, 0.78, 0.80])
        # ax = fig.add_axes([0.18, 0.21, 0.67, 0.70])
        ax.plot(
            zone_df['WY_PPT'].values, zone_df['WY_ETO'].values,
            linestyle='', marker='o', c=ncolors[1], ms=3, label='ETo')
        ax.plot(
            zone_df['WY_PPT'].values, zone_df['ET_MEAN'].values,
            linestyle='', marker='o', c=ncolors[2], ms=3, label='ET')
        # xmax = 100 * math.ceil(max(zone_df['WY_PPT']) / 100)
        # ymax = 200 * math.ceil((max(zone_df['WY_ETO']) + 200) / 200)
        ax.set_xlim([ppt_min, ppt_max])
        ax.set_ylim([0, 1.2 * max(zone_df['WY_ETO'])])
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
