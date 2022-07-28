#--------------------------------
# Name:         inputs.py
# Purpose:      Common INI reading/parsing functions
# Python:       3.6
#--------------------------------

from builtins import input
import datetime
import logging
import os
import sys

import configparser

import ee_tools.gdal_common as gdc
import ee_tools.utils as utils


def read(ini_path):
    logging.debug('\nReading Input File')
    # Open config file
    config = configparser.ConfigParser()
    try:
        config.read(ini_path)
    except Exception as e:
        logging.error('\nERROR: Input file could not be read, '
                      'is not an input file, or does not exist\n'
                      '  ini_path = {}\n{}\n'.format(ini_path, e))
        sys.exit()

    # Force conversion of unicode to strings
    ini = dict()
    for section in config.keys():
        ini[str(section)] = {}
        for k, v in config[section].items():
            ini[str(section)][str(k)] = v
    return ini


def parse_section(ini, section):
    logging.debug('Checking {} section'.format(section))
    if section not in ini.keys():
        logging.error(
            '\nERROR: Input file does not have an {} section'.format(section))
        sys.exit()

    if section == 'INPUTS':
        parse_inputs(ini)
    elif section == 'SPATIAL':
        parse_spatial_reference(ini)
    elif section == 'EXPORT':
        parse_export(ini)
    elif section == 'IMAGES':
        parse_images(ini)
    elif section == 'ZONAL_STATS':
        parse_zonal_stats(ini)
    elif section == 'SUMMARY':
        parse_summary(ini)
    elif section == 'FIGURES':
        parse_figures(ini)
    # elif section == 'TABLES':
    #     parse_tables(ini)
    elif section == 'BEAMER':
        parse_beamer(ini)
    elif section == 'GSHEET':
        parse_gsheet(ini)


def get_param(ini, section, input_name, output_name, get_type,
              default='MANDATORY'):
    """Get INI parameters by type and set default values

    Args:
        ini (dict): Nested dictionary of INI file keys/values
        section (str): Section name
        input_name (str): Parameter name in INI file
        output_name (str): Parameter name in code
        get_type (): Python type
        default (): Default value to use if parameter was not set.
            Defaults to "MANDATORY".
            "MANDATORY" will cause script to exit if key does not exist.
    """

    try:
        if get_type is bool:
            ini[section][output_name] = (
                ini[section][input_name].lower() == 'true')
            # ini[section][output_name] = distutils.util.strtobool(
            #     ini[section][input_name])
            # ini[section][output_name] = ini.getboolean(section, input_name)
            # ini[section][output_name] = ini[section].getboolean(input_name)
        elif get_type is int:
            ini[section][output_name] = int(ini[section][input_name])
        elif get_type is float:
            ini[section][output_name] = float(ini[section][input_name])
        elif get_type is list:
            ini[section][output_name] = str(ini[section][input_name])
            # Parsing strings to list is handled in each section separately
            # ini[section][output_name] = [
            #     x.strip()
            #     for x in str(ini[section][input_name]).split(',')]
        else:
            ini[section][output_name] = str(ini[section][input_name])
            # Convert 'None' (strings) to None
            if ini[section][output_name].lower() in ['none', '']:
                ini[section][output_name] = None
    except (KeyError, configparser.NoOptionError):
        if default == 'MANDATORY':
            logging.error(
                '\nERROR: {} was not set in the INI, exiting\n'.format(
                    input_name))
            sys.exit()
        else:
            ini[section][input_name] = default
            ini[section][output_name] = default
            logging.debug('  Setting {} = {}'.format(
                input_name, ini[section][output_name]))
    except ValueError:
        logging.error('\nERROR: Invalid value for "{}"'.format(
            input_name))
        sys.exit()
    except Exception as e:
        logging.error('\nERROR: Unhandled error\n  {}'.format(e))
        sys.exit()

    # If the parameter is renamed, remove the old name/parameter
    if input_name != output_name:
        del ini[section][input_name]


def parse_inputs(ini, section='INPUTS'):
    # MANDATORY PARAMETERS
    # section, input_name, output_name, description, get_type
    param_list = [
        ['zone_shp_path', 'zone_shp_path', str],
        ['zone_field', 'zone_field', str]
    ]
    for input_name, output_name, get_type in param_list:
        get_param(ini, section, input_name, output_name, get_type)

    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        # Control which Landsat images are used
        ['landsat4_flag', 'landsat4_flag', bool, False],
        ['landsat5_flag', 'landsat5_flag', bool, False],
        ['landsat7_flag', 'landsat7_flag', bool, False],
        ['landsat8_flag', 'landsat8_flag', bool, False],
        # Date filtering
        ['start_date', 'start_date', int, None],
        ['end_date', 'end_date', int, None],
        ['start_year', 'start_year', int, None],
        ['end_year', 'end_year', int, None],
        ['start_month', 'start_month', int, None],
        ['end_month', 'end_month', int, None],
        ['start_doy', 'start_doy', int, None],
        ['end_doy', 'end_doy', int, None],
        # Scene ID filtering
        ['scene_id_keep_path', 'scene_id_keep_path', str, ''],
        ['scene_id_skip_path', 'scene_id_skip_path', str, ''],
        # Tile filtering
        ['path_keep_list', 'path_keep_list', list, []],
        ['row_keep_list', 'row_keep_list', list, []],
        ['tile_keep_list', 'tile_keep_list', list, []],
        # FID filtering
        ['fid_skip_list', 'fid_skip_list', list, []],
        ['fid_keep_list', 'fid_keep_list', list, []],
        # Cloud masking
        ['acca_flag', 'acca_flag', bool, False],
        ['fmask_flag', 'fmask_flag', bool, False],
        # Default to Tasumi at-surface reflectance (for beamer calculation)
        ['refl_sur_method', 'refl_sur_method', str, 'tasumi'],
        ['mosaic_method', 'mosaic_method', str, 'mean'],
        ['adjust_method', 'adjust_method', str, None],
        ['merge_geometries_flag', 'merge_geom_flag', bool, False],
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # Strip off file extension from filename
    ini[section]['zone_filename'] = os.path.splitext(
        os.path.basename(ini[section]['zone_shp_path']))[0]

    # Zone shapefile
    # Eventually this may not be a mandatory parameter
    if not os.path.isdir(os.path.dirname(ini[section]['zone_shp_path'])):
        logging.error(
            '\nERROR: The zone workspace does not exist, exiting\n'
            '  {}'.format(os.path.dirname(ini[section]['zone_shp_path'])))
        sys.exit()
    elif not os.path.isfile(ini[section]['zone_shp_path']):
        logging.error(
            '\nERROR: The zone shapefile does not exist, '
            'exiting\n  {}'.format(ini[section]['zone_shp_path']))
        sys.exit()

    # Start/end year
    if (ini[section]['start_year'] and ini[section]['end_year'] and
            ini[section]['end_year'] < ini[section]['start_year']):
        logging.error(
            '\nERROR: End year must be >= start year')
        sys.exit()
    default_end_year = datetime.datetime.today().year + 1
    if ((ini[section]['start_year'] and
            ini[section]['start_year'] not in range(1984, default_end_year)) or
        (ini[section]['end_year'] and
            ini[section]['end_year'] not in range(1984, default_end_year))):
        logging.error('\nERROR: Year must be an integer from 1984-{}'.format(
            default_end_year - 1))
        sys.exit()

    # Start/end month
    if (ini[section]['start_month'] and
            ini[section]['start_month'] not in range(1, 13)):
        logging.error(
            '\nERROR: Start month must be an integer from 1-12')
        sys.exit()
    elif (ini[section]['end_month'] and
            ini[section]['end_month'] not in range(1, 13)):
        logging.error('\nERROR: End month must be an integer from 1-12')
        sys.exit()

    # Start/end DOY
    if ini[section]['end_doy'] and ini[section]['end_doy'] > 273:
        logging.error(
            '\nERROR: End DOY has to be in the same water year as start DOY')
        sys.exit()
    if (ini[section]['start_doy'] and
            ini[section]['start_doy'] not in range(1, 367)):
        logging.error(
            '\nERROR: Start DOY must be an integer from 1-366')
        sys.exit()
    elif (ini[section]['end_doy'] and
            ini[section]['end_doy'] not in range(1, 367)):
        logging.error('\nERROR: End DOY must be an integer from 1-366')
        sys.exit()
    # if ini[section]['end_doy'] < ini[section]['start_doy']:
    #     logging.error('\nERROR: End DOY must be >= start DOY')
    #     sys.exit()

    if ini[section]['fid_keep_list']:
        ini[section]['fid_keep_list'] = sorted(list(
            utils.parse_int_set(ini[section]['fid_keep_list'])))
    if ini[section]['fid_skip_list']:
        ini[section]['fid_skip_list'] = sorted(list(
            utils.parse_int_set(ini[section]['fid_skip_list'])))

    # Convert path/row ranges to list
    if ini[section]['path_keep_list']:
        ini[section]['path_keep_list'] = sorted(list(
            utils.parse_int_set(ini[section]['path_keep_list'])))
    if ini[section]['row_keep_list']:
        ini[section]['row_keep_list'] = sorted(list(
            utils.parse_int_set(ini[section]['row_keep_list'])))
    if ini[section]['tile_keep_list']:
        ini[section]['tile_keep_list'] = sorted([
            tile.strip()
            for tile in ini[section]['tile_keep_list'].split(',')])

    # Get list of path/row strings to centroid coordinates
    if ini[section]['tile_keep_list']:
        ini[section]['tile_geom'] = utils.wrs2_tile_geom_func(
            ini['INPUTS']['tile_keep_list'])
    else:
        ini[section]['tile_geom'] = None

    # Only process specific Landsat scenes
    ini[section]['scene_id_keep_list'] = []
    if ini[section]['scene_id_keep_path']:
        try:
            with open(ini[section]['scene_id_keep_path']) as input_f:
                scene_id_keep_list = input_f.readlines()
            ini[section]['scene_id_keep_list'] = [
                x.strip() for x in scene_id_keep_list]
        except IOError:
            logging.error('\nFileIO Error: {}'.format(
                ini[section]['scene_id_keep_path']))
            sys.exit()
        except Exception as e:
            logging.error('\nUnhanded Exception: {}'.format(e))

        # Parse full product ID down to GEE scene ID if necessary
        # LE07_L1TP_041035_20180127_20180222_01_T1
        ini[section]['scene_id_keep_list'] = [
            x[:5] + x[10:25] if len(x) == 40 else x
            for x in ini[section]['scene_id_keep_list']]

    # Skip specific landsat scenes
    ini[section]['scene_id_skip_list'] = []
    if ini[section]['scene_id_skip_path']:
        try:
            with open(ini[section]['scene_id_skip_path']) as input_f:
                scene_id_skip_list = input_f.readlines()
            ini[section]['scene_id_skip_list'] = [
                x.strip() for x in scene_id_skip_list]
        except IOError:
            logging.error('\nFileIO Error: {}'.format(
                ini[section]['scene_id_skip_path']))
            sys.exit()
        except Exception as e:
            logging.error('\nUnhanded Exception: {}'.format(e))

        # Parse full product ID down to GEE scene ID if necessary
        # LE07_L1TP_041035_20180127_20180222_01_T1
        ini[section]['scene_id_skip_list'] = [
            x[:5] + x[10:25] if len(x) == 40 else x
            for x in ini[section]['scene_id_skip_list']]

    # At-surface reflectance source type
    if ini[section]['refl_sur_method']:
        ini[section]['refl_sur_method'] = ini[section]['refl_sur_method'].lower()
        options = ['tasumi', 'usgs_sr', 'none']
        if ini[section]['refl_sur_method'] not in options:
            logging.error(
                '\nERROR: Invalid at-surface reflectance_method: {}\n'
                '  Must be: {}'.format(
                    ini[section]['refl_sur_method'], ', '.join(options)))
            sys.exit()

    # Mosaic method
    if ini[section]['mosaic_method']:
        ini[section]['mosaic_method'] = ini[section]['mosaic_method'].lower()
        options = ['mean', 'median', 'mosaic', 'min', 'max']
        if ini[section]['mosaic_method'] not in options:
            logging.error(
                '\nERROR: Invalid mosaic method: {}\n  Must be: {}'.format(
                    ini[section]['mosaic_method'], ', '.join(options)))
            sys.exit()

    # Adjust Landsat Red and NIR bands
    if ini[section]['adjust_method']:
        ini[section]['adjust_method'] = ini[section]['adjust_method'].lower()
        options = ['oli_2_etm', 'etm_2_oli']
        if ini[section]['adjust_method'] not in options:
            logging.error(
                '\nERROR: Invalid adjust method: {}\n  Must be: {}'.format(
                    ini[section]['adjust_method'], ', '.join(options)))
            sys.exit()


def parse_spatial_reference(ini, section='SPATIAL'):
    """"""
    # MANDATORY PARAMETERS
    # section, input_name, output_name, description, get_type
    param_list = [
        # Output spatial reference
        ['output_snap', 'snap', str],
        ['output_cs', 'cellsize', float],
        ['output_proj', 'crs', str]
    ]
    for input_name, output_name, get_type in param_list:
        get_param(ini, section, input_name, output_name, get_type)

    # Convert snap points to list
    ini[section]['snap'] = [
        float(i) for i in ini[section]['snap'].split(',')
        if i.strip().isdigit()][:2]
    # Compute snap points separately
    ini[section]['snap_x'], ini[section]['snap_y'] = ini[section]['snap']

    # Compute OSR from EGSG code
    try:
        ini[section]['osr'] = gdc.epsg_osr(
            int(ini[section]['crs'].split(':')[1]))
    except Exception as e:
        logging.error(
            '\nERROR: The output projection could not be converted to a '
            'spatial reference object\n  {}'.format(
                ini[section]['crs']))
        logging.exception('  {}'.format(e))
        sys.exit()

    logging.debug('  Snap: {} {}'.format(
        ini[section]['snap_x'], ini[section]['snap_y']))
    logging.debug('  Cellsize: {}'.format(ini[section]['cellsize']))
    logging.debug('  CRS: {}'.format(ini[section]['crs']))
    # logging.debug('  OSR: {}\n'.format(
    #     ini[section]['osr'].ExportToWkt())


def parse_export(ini, section='EXPORT'):
    """"""
    # MANDATORY PARAMETERS
    # section, input_name, output_name, description, get_type
    # param_list = [
    #     ['export_dest', 'export_dest', str]
    # ]
    # for input_name, output_name, get_type in param_list:
    #     get_param(ini, section, input_name, output_name, get_type)

    # DEADBEEF - Eventually switch this back to being a mandatory parameter
    param_list = [
        ['export_dest', 'export_dest', str, 'getinfo']
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # DEADBEEF - CLOUD export options are still experimental
    export_dest_options = ['gdrive', 'cloud', 'getinfo']
    ini[section]['export_dest'] = ini[section]['export_dest'].lower()
    if ini[section]['export_dest'] not in export_dest_options:
        logging.error(
            '\nERROR: Invalid Export Destination: {}\n  Must be {}'.format(
                ini[section]['export_dest'], ', '.join(export_dest_options)))
        sys.exit()

    # DEADBEEF - This might be better in an export module or separate function
    # Export destination specific options
    if ini[section]['export_dest'] in ['getinfo']:
        logging.info('  GetInfo Direct Export')
    elif ini[section]['export_dest'] in ['gdrive']:
        logging.info('  Google Drive Export')
        get_param(ini, section, 'gdrive_workspace', 'gdrive_ws', str)
        get_param(ini, section, 'export_folder', 'export_folder', str, '')

        # DEADBEEF
        if ini[section]['export_folder']:
            logging.info(
                '\nThere can be issues writing to Google Drive folders'
                '\nYou may want to clear the "export_folder" parameter in the INI')
            # ini[section]['export_folder'] = ''
            input('ENTER')
        elif not ini[section]['export_folder']:
            ini[section]['export_folder'] = ''

        # Build and check file paths
        ini[section]['export_ws'] = os.path.join(
            ini[section]['gdrive_ws'], ini[section]['export_folder'])
        if not os.path.isdir(ini[section]['export_ws']):
            os.makedirs(ini[section]['export_ws'])
        logging.debug('  {:16s} {}'.format(
            'GDrive Workspace:', ini['EXPORT']['export_ws']))
    elif ini[section]['export_dest'] == 'cloud':
        logging.info('  Cloud Storage')
        get_param(ini, section, 'project_name', 'project_name', str, 'steel-melody-531')
        get_param(ini, section, 'bucket_name', 'bucket_name', str, 'ee-tools-export')

        if not ini[section]['project_name']:
            logging.error('\nERROR: {} must be set in INI, exiting\n'.format(
                ini[section]['project_name']))
        elif ini[section]['project_name'] != 'steel-melody-531':
            logging.error(
                '\nERROR: When exporting to Cloud Storage, the ' +
                'project_name parameter sets the project name.' +
                '  This parameter must be set to "steel-melody-531" for now')
            sys.exit()
        if not ini[section]['bucket_name']:
            logging.error('\nERROR: {} must be set in INI, exiting\n'.format(
                ini[section]['bucket_name']))
            sys.exit()
        elif ini[section]['bucket_name'] != 'ee-tools-export':
            logging.error(
                '\nERROR: When exporting to Cloud Storage, the ' +
                'bucket_name parameter sets the project name.' +
                '  This parameter must be set to "ee-tools-export" for now')
            sys.exit()
        ini[section]['export_ws'] = 'gs://{}'.format(
            ini[section]['bucket_name'])
        logging.debug('  {:16s} {}'.format(
            'Project:', ini[section]['project_name']))
        logging.debug('  {:16s} {}'.format(
            'Bucket:', ini[section]['bucket_name']))
        logging.debug('  {:16s} {}'.format(
            'Cloud Workspace:', ini[section]['export_ws']))

        bucket_list = utils.get_buckets(ini[section]['project_name'])
        if ini[section]['bucket_name'] not in bucket_list:
            logging.error(
                '\nERROR: The bucket "{}" does not exist, exiting'.format(
                    ini[section]['bucket_name']))
            return False
            # Try creating the storage bucket if it doesn't exist using gsutil
            # For now, I think it is better to make the user go do this
            # subprocess.check_output([
            #     'gsutil', 'mb', '-p', ini[section]['project_name'],
            #     'gs://{}-{}'.format(
            #         ini[section]['project_name'],
            #         ini[section]['bucket_name'])])

    # OPTIONAL PARAMETERS
    # section, input_name, output_name, description, get_type, default
    param_list = [
        ['export_only', 'export_only', bool, False]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)


def parse_images(ini, section='IMAGES'):
    """"""
    # # Image download bands
    # get_param(ini, section, 'download_bands', 'download_bands', str)
    # ini[section]['download_bands'] = sorted(list(set(map(
    #     lambda x: x.strip().lower(),
    #     ini[section]['download_bands'].split(',')))))
    # logging.debug('  Output Bands:')
    # for band in ini[section]['download_bands']:
    #     logging.debug('    {}'.format(band))

    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['output_workspace', 'output_ws', str, os.getcwd()],
        ['download_bands', 'download_bands', str, ''],
        ['clip_landsat_flag', 'clip_landsat_flag', bool, True],
        ['image_buffer', 'image_buffer', int, 0],
        ['eto_units', 'eto_units', str, 'mm'],
        ['ppt_units', 'ppt_units', str, 'mm']
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # Image download bands
    ini[section]['download_bands'] = sorted(list(set(map(
        lambda x: x.strip().lower(),
        ini[section]['download_bands'].split(',')))))
    logging.debug('  Output Bands:')
    for band in ini[section]['download_bands']:
        logging.debug('    {}'.format(band))

    # Build output folder if necessary
    if not os.path.isdir(ini[section]['output_ws']):
        os.makedirs(ini[section]['output_ws'])


def parse_zonal_stats(ini, section='ZONAL_STATS'):
    """"""
    # Get the list of Landsat products to compute
    # DEADBEEF - What should the default Landsat products be?
    get_param(
        ini, section, 'landsat_products', 'landsat_products', list,
        'albedo_sur, evi_sur, ndvi_sur, ndvi_toa, ts')
    get_param(
        ini, section, 'gridmet_products', 'gridmet_products', list,
        'eto, ppt')

    # get_param(
    #     ini, section, 'landsat_products', 'landsat_products', list,
    #     'albedo_sur, ndvi_toa, ts')
    ini[section]['landsat_products'] = utils.unique_keep_order([
        x.lower().strip()
        for x in ini[section]['landsat_products'].split(',') if x.strip()])
    logging.debug('  Landsat Products:')
    for band in ini[section]['landsat_products']:
        logging.debug('    {}'.format(band))

    ini[section]['gridmet_products'] = utils.unique_keep_order([
        x.lower().strip()
        for x in ini[section]['gridmet_products'].split(',') if x.strip()])
    logging.debug('  GRIDMET Products:')
    for band in ini[section]['gridmet_products']:
        logging.debug('    {}'.format(band))

    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['output_workspace', 'output_ws', str, os.getcwd()],
        ['landsat_flag', 'landsat_flag', bool, True],
        ['gridmet_daily_flag', 'gridmet_daily_flag', bool, False],
        ['gridmet_monthly_flag', 'gridmet_monthly_flag', bool, False],
        ['pdsi_flag', 'pdsi_flag', bool, False],
        ['year_step', 'year_step', int, 1],
        ['zone_geojson', 'zone_geojson', str, ''],
        ['zone_tile_path', 'zone_tile_path', str, ''],
        # ['tile_scene_path', 'tile_scene_path', str, '']
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # Build output folder if necessary
    if not os.path.isdir(ini[section]['output_ws']):
        os.makedirs(ini[section]['output_ws'])

    # A year step of 60 will start on 1980
    if ini[section]['year_step'] < 1 or ini[section]['year_step'] > 60:
        logging.error('\nERROR: year_step must be an integer from 1-60')
        sys.exit()

    # Name zone-path/row-scene ID JSONs using the shapefile name if not set
    if not ini[section]['zone_geojson']:
        ini[section]['zone_geojson'] = ini['INPUTS']['zone_shp_path'] \
            .replace('.shp', '.geojson')
        logging.debug('  Setting {} = {}'.format(
            'zone_geojson', ini[section]['zone_geojson']))
    if not ini[section]['zone_tile_path']:
        ini[section]['zone_tile_path'] = ini['INPUTS']['zone_shp_path'] \
            .replace('.shp', '_tiles.json')
        logging.debug('  Setting {} = {}'.format(
            'zone_tile_path', ini[section]['zone_tile_path']))
    # if not ini[section]['tile_scene_path']:
    #     ini[section]['tile_scene_path'] = ini['INPUTS']['zone_shp_path'] \
    #         .replace('.shp', '_tiles.json')
    #     logging.debug('  Setting {} = {}'.format(
    #         'tile_scene_path', ini[section]['tile_scene_path']))

    # # Read in pre-computed path/row
    # if ini[section]['zone_tile_json']:
    #     try:
    #         with open(ini[section]['zone_tile_json']) as input_f:
    #             ini[section]['zone_tile_json'] = json.load(input_f)
    #     except IOError:
    #         logging.error('\nFileIO Error: {}'.format(
    #             ini[section]['zone_tile_json']))
    #         sys.exit()
    #     except Exception as e:
    #         logging.error('\nUnhanded Exception: {}'.format(e))
    # if ini[section]['tile_scene_json']:
    #     try:
    #         with open(ini[section]['tile_scene_json']) as input_f:
    #             ini[section]['tile_scene_json'] = json.load(input_f)
    #     except IOError:
    #         logging.error('\nFileIO Error: {}'.format(
    #             ini[section]['tile_scene_json']))
    #         sys.exit()
    #     except Exception as e:
    #         logging.error('\nUnhanded Exception: {}'.format(e))


def parse_summary(ini, section='SUMMARY'):
    """"""
    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['output_workspace', 'output_ws', str, os.getcwd()],
        # DEADBEEF - What should the default max_qa value be?
        ['max_qa', 'max_qa', float, 0],
        # Modified defaults to be consistent with example INI file and README
        ['max_cloud_score', 'max_cloud_score', float, 100],
        ['max_fmask_pct', 'max_fmask_pct', float, 100],
        ['min_slc_off_pct', 'min_slc_off_pct', float, 0],
        # Original default values
        # ['max_cloud_score', 'max_cloud_score', float, 70],
        # ['max_fmask_pct', 'max_fmask_pct', float, 100],
        # ['min_slc_off_pct', 'min_slc_off_pct', float, 50],
        ['gridmet_start_month', 'gridmet_start_month', int, 10],
        ['gridmet_end_month', 'gridmet_end_month', int, 9]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # Remove scenes with cloud score above target percentage
    if (ini[section]['max_cloud_score'] < 0 or
            ini[section]['max_cloud_score'] > 100):
        logging.error('\nERROR: max_cloud_score must be in the range 0-100')
        sys.exit()
    if (ini[section]['max_cloud_score'] > 0 and
            ini[section]['max_cloud_score'] < 1):
        logging.error(
            '\nWARNING: max_cloud_score must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        input('  Press ENTER to continue')

    # Remove scenes with Fmask counts above the target percentage
    if (ini[section]['max_fmask_pct'] < 0 or
            ini[section]['max_fmask_pct'] > 100):
        logging.error('\nERROR: max_fmask_pct must be in the range 0-100')
        sys.exit()
    if (ini[section]['max_fmask_pct'] > 0 and
            ini[section]['max_fmask_pct'] < 1):
        logging.error(
            '\nWARNING: max_fmask_pct must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        input('  Press ENTER to continue')

    # Remove SLC-off scenes with pixel counts below the target percentage
    if (ini[section]['min_slc_off_pct'] < 0 or
            ini[section]['min_slc_off_pct'] > 100):
        logging.error(
            '\nERROR: min_slc_off_pct must be in the range 0-100')
        sys.exit()
    if (ini[section]['min_slc_off_pct'] > 0 and
            ini[section]['min_slc_off_pct'] < 1):
        logging.error(
            '\nWARNING: min_slc_off_pct must be a percent (0-100)' +
            '\n  The value entered appears to be a decimal in the range 0-1')
        input('  Press ENTER to continue')

    # GRIDMET month range (default to water year)
    if (ini[section]['gridmet_start_month'] and
            ini[section]['gridmet_start_month'] not in range(1, 13)):
        logging.error(
            '\nERROR: GRIDMET start month must be an integer from 1-12')
        sys.exit()
    elif (ini[section]['gridmet_end_month'] and
            ini[section]['gridmet_end_month'] not in range(1, 13)):
        logging.error(
            '\nERROR: GRIDMET end month must be an integer from 1-12')
        sys.exit()
    if (ini[section]['gridmet_start_month'] is None and
            ini[section]['gridmet_end_month'] is None):
        ini[section]['gridmet_start_month'] = 10
        ini[section]['gridmet_end_month'] = 9


def parse_tables(ini, section='TABLES'):
    """"""
    # MANDATORY PARAMETERS
    # param_section, input_name, output_name, get_type
    param_list = [
        ['output_filename', 'output_filename', str]
    ]
    for input_name, output_name, get_type in param_list:
        get_param(ini, section, input_name, output_name, get_type)

    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['output_ws', 'output_ws', str, os.getcwd()]
        ['eto_units', 'eto_units', str, 'mm'],
        ['ppt_units', 'ppt_units', str, 'mm']
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    standardize_depth_units(ini, section, 'eto_units', 'ETo')
    standardize_depth_units(ini, section, 'ppt_units', 'PPT')


def parse_figures(ini, section='FIGURES'):
    """"""
    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['output_ws', 'output_ws', str, os.getcwd()],
        ['eto_units', 'eto_units', str, 'mm'],
        ['ppt_units', 'ppt_units', str, 'mm'],
        ['ppt_plot_type', 'ppt_plot_type', str, 'LINE'],
        ['best_fit_flag', 'scatter_best_fit', bool, False],
        ['figure_bands', 'figure_bands', str, 'ndvi_toa'],
        ['scatter_bands', 'scatter_bands', str, 'ppt:ndvi_sur, ppt:evi_sur'],
        ['complementary_bands', 'complementary_bands', str, 'evi_sur'],
        # Bokeh timeseries figures
        ['timeseries_bands', 'timeseries_bands', str, 'ndvi_toa, albedo_sur, ts']
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    standardize_depth_units(ini, section, 'eto_units', 'ETo')
    standardize_depth_units(ini, section, 'ppt_units', 'PPT')

    ini[section]['figure_bands'] = list(map(
        lambda x: x.strip().lower(),
        ini[section]['figure_bands'].split(',')))
    ini[section]['scatter_bands'] = [
        list(map(lambda x: x.strip().lower(), b.split(':')))
        for b in ini[section]['scatter_bands'].split(',')]
    ini[section]['complementary_bands'] = list(map(
        lambda x: x.strip().lower(),
        ini[section]['complementary_bands'].split(',')))
    ini[section]['timeseries_bands'] = list(map(
        lambda x: x.strip().lower(),
        ini[section]['timeseries_bands'].split(',')))

    if ini[section]['ppt_plot_type'].upper() not in ['LINE', 'BAR']:
        logging.error('\nERROR: ppt_plot_type must be "LINE" or "BAR"')
        sys.exit()


def parse_beamer(ini, section='BEAMER'):
    """"""
    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['output_name', 'output_name', str, ''],
        ['month_step', 'month_step', int, 1],
        ['eto_source', 'eto_source', str, 'GRIDMET'],
        ['eto_factor', 'eto_factor', float, 1.0],
        ['ppt_source', 'ppt_source', str, 'GRIDMET'],
        ['ppt_factor', 'ppt_factor', float, 1.0],
        ['data_eto_units', 'data_eto_units', str, 'mm'],
        ['data_ppt_units', 'data_ppt_units', str, 'mm'],
        ['eto_units', 'eto_units', str, 'mm'],
        ['ppt_units', 'ppt_units', str, 'mm'],
        ['etstar_threshold', 'etstar_threshold', float, 0]
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)

    # if (ini[section]['output_name'] and
    #         not ini[section]['output_name'].endswith('.csv')):
    #     logging.error('\nERROR: BEAMER output name must be a CSV type')
    #    sys.exit()

    # Beamer can't compute more than one year per iteration
    #   since it is dependent of water year ETo & PPT
    if ini[section]['month_step'] < 1 or ini[section]['month_step'] > 12:
        logging.error(
            '\nERROR: BEAMER month_step must be an integer from 1-12')
        sys.exit()

    if ini[section]['etstar_threshold'] < 0:
        logging.error(
            '\nERROR: BEAMER etstar_threshold must be >= 0')
        sys.exit()

    # Force inputs to lower case
    ini[section]['eto_source'] = ini[section]['eto_source'].lower()
    ini[section]['ppt_source'] = ini[section]['ppt_source'].lower()
    ini[section]['data_eto_units'] = ini[section]['data_eto_units'].lower()
    ini[section]['data_ppt_units'] = ini[section]['data_ppt_units'].lower()
    ini[section]['eto_units'] = ini[section]['eto_units'].lower()
    ini[section]['ppt_units'] = ini[section]['ppt_units'].lower()

    # ETo and PPT source
    eto_source_options = ['gridmet', 'file']
    ppt_source_options = ['gridmet', 'file']
    # ppt_source_list = ['file', 'gridmet', 'prism']

    if ini[section]['eto_source'] not in eto_source_options:
        logging.error(
            '\nERROR: Invalid eto_source: {}\n  Must be {}'.format(
                ini[section]['eto_source'], ', '.join(eto_source_options)))
        sys.exit()
    if ini[section]['ppt_source'] not in ppt_source_options:
        logging.error(
            '\nERROR: Invalid ppt_source: {}\n  Must be {}'.format(
                ini[section]['ppt_source'], ', '.join(ppt_source_options)))
        sys.exit()

    # Get data file path and fields if reading from file
    if (ini[section]['eto_source'] == 'file' or
            ini[section]['ppt_source'] == 'file'):
        get_param(ini, section, 'data_path', 'data_path', str)
        if not os.path.isfile(ini[section]['data_path']):
            logging.error(
                '\nERROR: The data_path does not exist\n  {}'.format(
                    ini[section]['data_path']))
            sys.exit()
        get_param(ini, section, 'data_zone_field', 'data_zone_field', str)
        get_param(ini, section, 'data_year_field', 'data_year_field', str)

    # Get ETo/PPT specific fields and input units if reading from file
    if ini[section]['eto_source'] == 'file':
        get_param(ini, section, 'data_eto_field', 'data_eto_field', str)
    if ini[section]['ppt_source'] == 'file':
        get_param(ini, section, 'data_ppt_field', 'data_ppt_field', str)

    # Overwrite input units if using known gridded data sources
    if ini[section]['eto_source'] in ['gridmet']:
        ini[section]['data_eto_units'] = 'mm'
    if ini[section]['ppt_source'] in ['gridmet']:
        ini[section]['data_ppt_units'] = 'mm'

    standardize_depth_units(ini, section, 'data_eto_units', 'ETo')
    standardize_depth_units(ini, section, 'data_ppt_units', 'PPT')
    standardize_depth_units(ini, section, 'eto_units', 'ETo')
    standardize_depth_units(ini, section, 'ppt_units', 'PPT')


def parse_gsheet(ini, section='GSHEET'):
    """"""
    # MANDATORY PARAMETERS
    # param_section, input_name, output_name, get_type
    param_list = [
        ['gsheet_id', 'gsheet_id', str]
    ]
    for input_name, output_name, get_type in param_list:
        get_param(ini, section, input_name, output_name, get_type)

    # OPTIONAL PARAMETERS
    # param_section, input_name, output_name, get_type, default
    param_list = [
        ['landsat_daily', 'landsat_daily', str, 'Landsat_Daily'],
        ['gridmet_daily', 'gridmet_daily', str, 'GRIDMET_Daily'],
        ['gridmet_monthly', 'gridmet_monthly', str, 'GRIDMET_Monthly']
    ]
    for input_name, output_name, get_type, default in param_list:
        get_param(ini, section, input_name, output_name, get_type, default)


def standardize_depth_units(ini, section, param, name):
    """"""
    ini[section][param] = ini[section]['eto_units'].lower()

    # Standardize unit naming
    units_remap = {'inches': 'in', 'feet': 'ft'}
    if ini[section]['eto_units'] in units_remap.keys():
        ini[section]['eto_units'] = units_remap[
            ini[section]['eto_units']]

    # Check input and output units
    unit_options = ['mm', 'ft', 'in']
    if ini[section][param] not in unit_options:
        logging.error(
            ('\nERROR: The {} {} units {} are invalid'
             '\n  Please set units to: {}').format(
                section, name, ini[section][param],
                ', '.join(unit_options)))
        sys.exit()
