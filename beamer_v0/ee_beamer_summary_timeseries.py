#--------------------------------
# Name:         ee_beamer_summary_timeseries.py
# Purpose:      Generate interactive timeseries figures
# Created       2017-07-27
# Python:       3.6
#--------------------------------

import argparse
from builtins import input
import datetime
import logging
import os
import sys

from bokeh.io import output_file, save, show
from bokeh.layouts import gridplot
import bokeh.models
from bokeh.models.glyphs import Circle
from bokeh.plotting import figure
import matplotlib as mpl
import matplotlib.cm as cm
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
    """Generate Bokeh figures

    Bokeh issues:
    Adjust y range based on non-muted data
        https://stackoverflow.com/questions/43620837/how-to-get-bokeh-to-dynamically-adjust-y-range-when-panning
    Linked interactive legends so that there is only one legend for the gridplot
    Maybe hide or mute QA values above max (instead of filtering them in advance)

    Args:
        ini_path (str):
        show_flag (bool): if True, show the figures in the browser.
            Default is False.
        overwrite_flag (bool): if True, overwrite existing tables.
            Default is True (for now)
    """
    logging.info('\nGenerate interactive timeseries figures')

    # Eventually read from INI
    plot_var_list = ['NDVI_TOA', 'ALBEDO_SUR', 'TS', 'NDWI_TOA', 'EVI_SUR']
    # plot_var_list = [
    #     'NDVI_TOA', 'ALBEDO_SUR', 'TS', 'NDWI_TOA',
    #     'CLOUD_SCORE', 'FMASK_PCT']
    output_folder = 'figures'

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
        # The names are currently stored in the CSV as spaces
        zone_output_name = zone_name.replace(' ', '_')
        zone_df = input_df[input_df['ZONE_NAME'] == zone_name]
        if zone_df.empty:
            logging.info('  Empty zone dataframe, skipping zone')
            continue

        # Output file paths
        output_doy_path = os.path.join(
            output_ws, '{}_timeseries_doy.html'.format(zone_output_name))
        output_date_path = os.path.join(
            output_ws, '{}_timeseries_date.html'.format(zone_output_name))

        # # Check for QA field
        # if 'QA' not in zone_df.columns.values:
        #     # logging.warning(
        #     #     '  WARNING: QA field not present in CSV\n'
        #     #     '  To compute QA/QC values, please run "ee_summary_qaqc.py"\n'
        #     #     '  Script will continue with no QA/QC values')
        #     zone_df['QA'] = 0
        #     # raw_input('ENTER')
        #     # logging.error(
        #     #     '\nPlease run the "ee_summary_qaqc.py" script '
        #     #     'to compute QA/QC values\n')
        #     # sys.exit()

        # Check that plot variables are present
        for plot_var in plot_var_list:
            if plot_var not in zone_df.columns.values:
                logging.error(
                    '  The variable {} does not exist in the '
                    'dataframe'.format(plot_var))
                sys.exit()

        # if ini['INPUTS']['scene_id_keep_list']:
        #     # Replace XXX with primary ROW value for checking skip list SCENE_ID
        #     scene_id_df = pd.Series([
        #         s.replace('XXX', '{:03d}'.format(int(r)))
        #         for s, r in zip(zone_df['SCENE_ID'], zone_df['ROW'])])
        #     zone_df = zone_df[scene_id_df.isin(
        #         ini['INPUTS']['scene_id_keep_list']).values]
        #     # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
        #     # zone_df = zone_df[zone_df['SCENE_ID'].isin(
        #     #     ini['INPUTS']['scene_id_keep_list'])]
        # if ini['INPUTS']['scene_id_skip_list']:
        #     # Replace XXX with primary ROW value for checking skip list SCENE_ID
        #     scene_id_df = pd.Series([
        #         s.replace('XXX', '{:03d}'.format(int(r)))
        #         for s, r in zip(zone_df['SCENE_ID'], zone_df['ROW'])])
        #     zone_df = zone_df[np.logical_not(scene_id_df.isin(
        #         ini['INPUTS']['scene_id_skip_list']).values)]
        #     # This won't work: SCENE_ID have XXX but scene_id_skip_list don't
        #     # zone_df = zone_df[np.logical_not(zone_df['SCENE_ID'].isin(
        #     #     ini['INPUTS']['scene_id_skip_list']))]

        # Compute colors for each QA value
        logging.debug('  Building column data source')
        qa_values = sorted(list(set(zone_df['QA'].values)))
        colors = {
            qa: "#%02x%02x%02x" % (int(r), int(g), int(b))
            for qa, (r, g, b, _) in zip(
                qa_values,
                255 * cm.viridis(mpl.colors.Normalize()(qa_values)))
        }
        logging.debug('  QA values: {}'.format(
            ', '.join(map(str, qa_values))))

        # Unpack the data by QA type to support interactive legends
        sources = dict()
        for qa_value in qa_values:
            qa_df = zone_df[zone_df['QA'] == qa_value]
            qa_data = {
                'INDEX': list(range(len(qa_df.index))),
                'PLATFORM': qa_df['PLATFORM'],
                'DATE': pd.to_datetime(qa_df['DATE']),
                'DATE_STR': pd.to_datetime(qa_df['DATE']).map(
                    lambda x: x.strftime('%Y-%m-%d')),
                'DOY': qa_df['DOY'].values,
                'QA': qa_df['QA'].values,
                'COLOR': [colors[qa] for qa in qa_df['QA'].values]
            }
            for plot_var in plot_var_list:
                if plot_var in qa_df.columns.values:
                    qa_data.update({plot_var: qa_df[plot_var].values})
            sources[qa_value] = bokeh.models.ColumnDataSource(qa_data)

        tooltips = [
            ("LANDSAT", "@PLATFORM"),
            ("DATE", "@TIME"),
            ("DOY", "@DOY")]

        # Selection
        hover_circle = Circle(
            fill_color='#ff0000', line_color='#ff0000')
        selected_circle = Circle(
            fill_color='COLOR', line_color='COLOR')
        nonselected_circle = Circle(
            fill_color='#aaaaaa', line_color='#aaaaaa')


        # Plot the data by DOY
        logging.debug('  Building DOY timeseries figure')
        if os.path.isfile(output_doy_path):
            os.remove(output_doy_path)
        output_file(output_doy_path, title=zone_name)

        figure_args = dict(
            plot_width=750, plot_height=250, title=None,
            tools="xwheel_zoom,xpan,xbox_zoom,reset,box_select",
            # tools="xwheel_zoom,xpan,xbox_zoom,reset,tap",
            active_scroll="xwheel_zoom")
        plot_args = dict(
            size=4, alpha=0.9, color='COLOR')
        if ini['SUMMARY']['max_qa'] > 0:
            plot_args['legend'] = 'QA'

        figures = []
        for plot_i, plot_var in enumerate(plot_var_list):
            if plot_i == 0:
                f = figure(
                    # x_range=Range1d(1, 366, bounds=(1, 366)),
                    y_axis_label=plot_var, **figure_args)
            else:
                f = figure(
                    x_range=f.x_range, y_axis_label=plot_var, **figure_args)

            for qa, source in sorted(sources.items()):
                r = f.circle('DOY', plot_var, source=source, **plot_args)
                r.hover_glyph = hover_circle
                r.selection_glyph = selected_circle
                r.nonselection_glyph = nonselected_circle
                r.muted_glyph = nonselected_circle

                # DEADBEEF - This will display high QA points as muted
                # if qa > ini['SUMMARY']['max_qa']:
                #     r.muted = True
                #     # r.visible = False

            f.add_tools(bokeh.models.HoverTool(tooltips=tooltips))

            # if ini['SUMMARY']['max_qa'] > 0:
            f.legend.location = "top_left"
            f.legend.click_policy = "hide"
            # f.legend.click_policy = "mute"
            f.legend.orientation = "horizontal"

            figures.append(f)

        # Try to not allow more than 4 plots in a column
        p = gridplot(
            figures, ncols=len(plot_var_list) // 3,
            sizing_mode='stretch_both')

        if show_flag:
            show(p)
        save(p)


        # Plot the data by DATE
        logging.debug('  Building date timeseries figure')
        if os.path.isfile(output_date_path):
            os.remove(output_date_path)
        output_file(output_date_path, title=zone_name)

        figure_args = dict(
            plot_width=750, plot_height=250, title=None,
            tools="xwheel_zoom,xpan,xbox_zoom,reset,box_select",
            # tools="xwheel_zoom,xpan,xbox_zoom,reset,tap",
            active_scroll="xwheel_zoom",
            x_axis_type="datetime",)
        plot_args = dict(
            size=4, alpha=0.9, color='COLOR')
        if ini['SUMMARY']['max_qa'] > 0:
            plot_args['legend'] = 'QA'

        figures = []
        for plot_i, plot_var in enumerate(plot_var_list):
            if plot_i == 0:
                f = figure(
                    # x_range=Range1d(x_limit[0], x_limit[1], bounds=x_limit),
                    y_axis_label=plot_var, **figure_args)
            else:
                f = figure(
                    x_range=f.x_range, y_axis_label=plot_var, **figure_args)

            if plot_var == 'TS':
                f.y_range.bounds = (270, None)

            for qa, source in sorted(sources.items()):
                r = f.circle('DATE', plot_var, source=source, **plot_args)
                r.hover_glyph = hover_circle
                r.selection_glyph = selected_circle
                r.nonselection_glyph = nonselected_circle
                r.muted_glyph = nonselected_circle

                # DEADBEEF - This will display high QA points as muted
                # if qa > ini['SUMMARY']['max_qa']:
                #     r.muted = True
                #     # r.visible = False

            f.add_tools(bokeh.models.HoverTool(tooltips=tooltips))

            # if ini['SUMMARY']['max_qa'] > 0:
            f.legend.location = "top_left"
            f.legend.click_policy = "hide"
            # f.legend.click_policy = "mute"
            f.legend.orientation = "horizontal"

            figures.append(f)

        # Try to not allow more than 4 plots in a column
        p = gridplot(
            figures, ncols=len(plot_var_list) // 3,
            sizing_mode='stretch_both')

        if show_flag:
            show(p)
        save(p)

        # Pause after each iteration if show is True
        if show_flag:
            input('Press ENTER to continue')


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate interactive timeseries figures',
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
