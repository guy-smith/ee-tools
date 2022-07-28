#--------------------------------
# Name:         ee_eddi_image_download.py
# Purpose:      Earth Engine EDDI Image Download
# Python:       3.6
#--------------------------------

import argparse
import datetime
import json
import logging
import os
import shutil
import sys

import ee
from osgeo import ogr

# import ee_tools.ee_common as ee_common
import ee_tools.gdal_common as gdc
import ee_tools.inputs as inputs
import ee_tools.utils as utils


def ee_image_download(ini_path=None, overwrite_flag=False):
    """Earth Engine Annual Mean Image Download

    Parameters
    ----------
    ini_path : str
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).

    """
    logging.info('\nEarth Engine EDDI Image Download')

    # 12 month EDDI
    aggregation_days = 365
    export_name = 'eddi_12month'
    output_name = 'eddi.12month'

    eddi_date_list = [
        '0131', '0228', '0331', '0430', '0531', '0630',
        '0731', '0831', '0930', '1031', '1130', '1231']
    # eddi_date_list = ['0930', '1231']
    # eddi_date_list = ['{:02d}01'.format(m) for m in range(1, 13)]
    # eddi_date_list = []

    eddi_folder = 'eddi'

    # Do we need to support separate EDDI years?
    # start_year = 1984
    # end_year = 2016

    #
    climo_year_start = 1979
    climo_year_end = 2017

    # Read config file
    # ini = inputs.ini_parse(ini_path, section='IMAGE')
    ini = inputs.read(ini_path)
    inputs.parse_section(ini, section='INPUTS')
    inputs.parse_section(ini, section='SPATIAL')
    inputs.parse_section(ini, section='EXPORT')
    inputs.parse_section(ini, section='IMAGES')

    nodata_value = -9999

    # Manually set output spatial reference
    logging.info('\nHardcoding GRIDMET snap, cellsize and spatial reference')
    ini['output_x'], ini['output_y'] = -124.79299639209513, 49.41685579737572
    ini['SPATIAL']['cellsize'] = 0.041666001963701
    # ini['SPATIAL']['cellsize'] = [0.041666001963701, 0.041666001489718]
    # ini['output_x'] = -124.79166666666666666667
    # ini['output_y'] = 25.04166666666666666667
    # ini['SPATIAL']['cellsize'] = 1. / 24
    ini['SPATIAL']['osr'] = gdc.epsg_osr(4326)
    # ini['SPATIAL']['osr'] = gdc.epsg_osr(4269)
    ini['SPATIAL']['crs'] = 'EPSG:4326'
    logging.debug('  Snap: {} {}'.format(ini['output_x'], ini['output_y']))
    logging.debug('  Cellsize: {}'.format(ini['SPATIAL']['cellsize']))
    logging.debug('  OSR: {}'.format(ini['SPATIAL']['osr']))

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

    # Merge geometries
    if ini['INPUTS']['merge_geom_flag']:
        merge_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        for zone in zone_geom_list:
            zone_multipolygon = ogr.ForceToMultiPolygon(
                ogr.CreateGeometryFromJson(json.dumps(zone[2])))
            for zone_polygon in zone_multipolygon:
                merge_geom.AddGeometry(zone_polygon)
        # merge_json = json.loads(merge_mp.ExportToJson())
        zone_geom_list = [[
            0, ini['INPUTS']['zone_filename'],
            json.loads(merge_geom.ExportToJson())]]
        ini['INPUTS']['zone_field'] = ''

    # Need zone_shp_path projection to build EE geometries
    zone_osr = gdc.feature_path_osr(ini['INPUTS']['zone_shp_path'])
    zone_proj = gdc.osr_wkt(zone_osr)
    # zone_proj = ee.Projection(zone_proj).wkt().getInfo()
    # zone_proj = zone_proj.replace('\n', '').replace(' ', '')
    logging.debug('  Zone Projection: {}'.format(zone_proj))


    # Initialize Earth Engine API key
    logging.info('\nInitializing Earth Engine')
    ee.Initialize()
    utils.ee_request(ee.Number(1).getInfo())

    # Get current running tasks
    tasks = utils.get_ee_tasks()


    # Download images for each feature separately
    for zone_fid, zone_name, zone_json in zone_geom_list:
        zone_name = zone_name.replace(' ', '_')
        logging.info('ZONE: {} (FID: {})'.format(zone_name, zone_fid))

        # Build EE geometry object for zonal stats
        zone_geom = ee.Geometry(zone_json, zone_proj, False)

        # Project the zone_geom to the GRIDMET projection
        # if zone_proj != output_proj:
        zone_geom = zone_geom.transform(ini['SPATIAL']['crs'], 0.001)

        # Get the extent from the Earth Engine geometry object?
        zone_extent = zone_geom.bounds().getInfo()['coordinates'][0]
        zone_extent = gdc.Extent([
            min(zip(*zone_extent)[0]), min(zip(*zone_extent)[1]),
            max(zip(*zone_extent)[0]), max(zip(*zone_extent)[1])])
        # # Use GDAL and geometry json to build extent, transform, and shape
        # zone_extent = gdc.Extent(
        #     ogr.CreateGeometryFromJson(json.dumps(zone_json)).GetEnvelope())
        # # zone_extent = gdc.Extent(zone_geom.GetEnvelope())
        # zone_extent.ymin, zone_extent.xmax = zone_extent.xmax, zone_extent.ymin

        # Adjust extent to match raster
        zone_extent = zone_extent.adjust_to_snap(
            'EXPAND', ini['output_x'], ini['output_y'],
            ini['SPATIAL']['cellsize'])
        zone_geo = zone_extent.geo(ini['SPATIAL']['cellsize'])
        zone_transform = gdc.geo_2_ee_transform(zone_geo)
        zone_transform = '[' + ','.join(map(str, zone_transform)) + ']'
        zone_shape = zone_extent.shape(ini['SPATIAL']['cellsize'])
        logging.debug('  Zone Shape: {}'.format(zone_shape))
        logging.debug('  Zone Transform: {}'.format(zone_transform))
        logging.debug('  Zone Extent: {}'.format(zone_extent))
        # logging.debug('  Geom: {}'.format(zone_geom.getInfo()))

        # output_transform = zone_transform[:]
        output_transform = '[' + ','.join(map(str, zone_transform)) + ']'
        output_shape = '[{1}x{0}]'.format(*zone_shape)
        logging.debug('  Output Projection: {}'.format(ini['SPATIAL']['crs']))
        logging.debug('  Output Transform: {}'.format(output_transform))
        logging.debug('  Output Shape: {}'.format(output_shape))

        zone_eddi_ws = os.path.join(
            ini['IMAGES']['output_ws'], zone_name, eddi_folder)
        if not os.path.isdir(zone_eddi_ws):
            os.makedirs(zone_eddi_ws)

        # GRIDMET PDSI
        # Process each image in the collection by date
        export_list = []

        export_list = list(date_range(
            datetime.datetime(ini['INPUTS']['start_year'], 1, 1),
            datetime.datetime(ini['INPUTS']['end_year'], 12, 31),
            skip_leap_days=True))

        # Filter list to only keep last dekad of October and December
        if eddi_date_list:
            export_list = [
                tgt_dt for tgt_dt in export_list
                if tgt_dt.strftime('%m%d') in eddi_date_list]

        for tgt_dt in export_list:
            date_str = tgt_dt.strftime('%Y%m%d')
            logging.info('{} {}'.format(
                tgt_dt.strftime('%Y-%m-%d'), output_name))

            if tgt_dt >= datetime.datetime.today():
                logging.info('  Date after current date, skipping')
                continue

            # Rename to match naming style from getDownloadURL
            #     image_name.band.tif
            export_id = '{}_{}_{}'.format(
                ini['INPUTS']['zone_filename'], date_str, export_name.lower())
            output_id = '{}_{}'.format(date_str, output_name)

            export_path = os.path.join(
                ini['EXPORT']['export_ws'], export_id + '.tif')
            output_path = os.path.join(
                zone_eddi_ws, output_id + '.tif')
            logging.debug('  Export: {}'.format(export_path))
            logging.debug('  Output: {}'.format(output_path))

            if overwrite_flag:
                if export_id in tasks.keys():
                    logging.debug('  Task already submitted, cancelling')
                    ee.data.cancelTask(tasks[export_id])
                    del tasks[export_id]
                if os.path.isfile(export_path):
                    logging.debug('  Export image already exists, removing')
                    utils.remove_file(export_path)
                    # os.remove(export_path)
                if os.path.isfile(output_path):
                    logging.debug('  Output image already exists, removing')
                    utils.remove_file(output_path)
                    # os.remove(output_path)
            else:
                if os.path.isfile(export_path):
                    logging.debug('  Export image already exists, moving')
                    shutil.move(export_path, output_path)
                    gdc.raster_path_set_nodata(output_path, nodata_value)
                    # DEADBEEF - should raster stats be computed?
                    # gdc.raster_statistics(output_path)
                    continue
                elif os.path.isfile(output_path):
                    logging.debug('  Output image already exists, skipping')
                    continue
                elif export_id in tasks.keys():
                    logging.debug('  Task already submitted, skipping')
                    continue

            eddi_image = ee_eddi_image(
                tgt_dt.strftime('%Y-%m-%d'), agg_days=aggregation_days,
                variable='eddi',
                year_start=climo_year_start, year_end=climo_year_end)

            logging.debug('  Building export task')
            # if ini['EXPORT']['export_dest'] == 'gdrive':
            task = ee.batch.Export.image.toDrive(
                image=eddi_image,
                description=export_id,
                folder=ini['EXPORT']['export_folder'],
                fileNamePrefix=export_id,
                dimensions=output_shape,
                crs=ini['SPATIAL']['crs'],
                crsTransform=output_transform)
            # elif ini['EXPORT']['export_dest'] == 'gdrive':
            #     task = ee.batch.Export.image.toCloudStorage(
            #         image=eddi_image,
            #         description=export_id,
            #         bucket=ini['EXPORT']['export_folder'],
            #         fileNamePrefix=export_id,
            #         dimensions=output_shape,
            #         crs=ini['SPATIAL']['crs'],
            #         crsTransform=output_transform)

            logging.debug('  Starting export task')
            utils.ee_request(task.start())


def ee_eddi_image(tgt_date, agg_days=30, variable='eddi',
                  year_start=1979, year_end=datetime.datetime.today().year):
    """Generage EE EDDI images

    Args:
        tgt_date (str): date to process (in ISO format: YYYY-MM-DD)
        agg_days (int): numbers of days in aggregation period
        variable (str):
        year_start (int): climatology start year
        year_end (int): climatology end year

    Returns:
        ee.Image
    """

    # Derive band names from variable
    if variable == 'eddi':
        band = 'eto'
    elif variable == 'spi':
        band = 'pr'
    # elif variable == 'spei':
    #     band = '?'
    else:
        return 'Invalid variable'

    # Compute normalized probablity image
    output_image = ee_normprob_func(
        datetime.datetime.strptime(tgt_date, '%Y-%m-%d'),
        agg_days, band, year_start, year_end)
    return output_image


def ee_normprob_func(tgt_dt, agg_days=30, band='eto',
                     year_start=1979, year_end=datetime.datetime.today().year):
    """Compute normalized probability image for a single date using EE

    eto (grass reference ET) -> EDDI
    pr (precipitation) -> SPI

    Args:
        tgt_dt (datetime): end date
        agg_days (int): numbers of days in aggregation period
        band (str): Earth Engine band name
        year_start (int): climatology start year
        year_end (int): climatology end year

    Returns:
        Earth Engine image
    """

    # Unpack the date string
    tgt_year = tgt_dt.year
    tgt_month = tgt_dt.month
    tgt_day = tgt_dt.day

    # Adjust start year if there are not enough days to aggregate
    # Compute "DOY" for the date but for a non-leap year
    if int(datetime.datetime(1979, tgt_month, tgt_day).strftime('%j')) < agg_days:
        year_start += 1

    # Build the climatology
    year_list = ee.List.sequence(year_start, year_end)
    years = year_end - year_start + 1

    # For other collections, it may be neccesary to compute the target variable
    tgt_coll = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').select([band])

    # To match Dan's calculations, remove lap days from collection
    leap_day_filter = ee.Filter.calendarRange(2, 2, 'month')\
        .And(ee.Filter.calendarRange(29, 29, 'day_of_month'))\
        .Not()
    tgt_coll = tgt_coll.filter(leap_day_filter)

    def year_sum_func(year):
        # Compute start/end date for a non-leap year (to get correct difference)
        year_offset = ee.Number(year).subtract(1979)
        # Place end date 1 day past target DOY
        # Since filterDate is not inclusive on end date and dates are 0:00 UTC
        end_date = ee.Date.fromYMD(1979, tgt_month, tgt_day).advance(1, 'day')
        start_date = end_date.advance(-agg_days, 'day')
        # Advance start and end date back to target year
        end_date = end_date.advance(year_offset, 'year')
        start_date = start_date.advance(year_offset, 'year')

        # Original method
        # end_date = ee.Date.fromYMD(
        #     year, tgt_month, tgt_day).advance(1, 'day')
        # start_date = end_date.advance(-agg_days, 'day')

        # Set start/end times to target DOY
        return ee.Image(tgt_coll.filterDate(start_date, end_date).sum()) \
            .set({
                # 'DOY':tgt_doy, 'YEAR':year,
                'system:time_start': start_date.millis(),
                'system:time_end': end_date.advance(-1, 'second').millis()})
        # return gridmet_coll.filterDate(start_date, end_date)
    tgt_coll = ee.ImageCollection(year_list.map(year_sum_func))

    # If number of images/years is known and value is in climatology
    # Sorted position/index can be found by constructing percentiles for each image
    # If there were 30 years in the climo, construct 30 percentile steps from 0 and 100
    # (year - 1 to go from edges to steps)
    pct_step = 100.0 / (years - 1)
    pct_list = ee.List.sequence(0, 100 - pct_step, pct_step)
    # print(pct_step)
    # print(pct_list)

    # Calculate the values at the percentiles
    tgt_percentiles = tgt_coll.reduce(ee.Reducer.percentile(pct_list))

    # Compare target ETo image to climos
    tgt_image = year_sum_func(tgt_year)
    # Map.addLayer(tgt_image, {min:0, max:10}, tgt_year.toString())

    # Get 1's below, 0's above.. a mean will give the percent below value.
    positions = tgt_image.gt(tgt_percentiles).reduce(ee.Reducer.sum())

    # Compute Tukey plotting positions
    # Add 1 since positions were computed as 0 based indices
    # github.com/statsmodels/statsmodels/blob/master/statsmodels/sandbox/stats/stats_mstats_short.py
    alpha = 1.0 / 3
    # beta = 1.0 / 3
    pp = positions.float().add(1 - alpha).divide(years + alpha)

    # Compute from inverse CDF of plotting positions
    # Following Abramowitz and Stegun (1965) outlined:
    # http:# journals.ametsoc.org/doi/pdf/10.1175/2009JCLI2909.1
    p = pp.multiply(-1).add(1)
    # mask0 = p.lte(0.5)
    mask1 = p.gt(0.5)

    w0 = p.log().multiply(-2).sqrt()
    w1 = pp.log().multiply(-2).sqrt()

    ppf_expr = (
        'w - (2.515517 + 0.802853 * w + 0.010328 * w ** 2) / '
        '(1 + 1.432788 * w + 0.189269 * w ** 2 + 0.001308 * w ** 3)')
    output0 = ee.Image(w0.expression(ppf_expr, {'w': w0}))
    output1 = ee.Image(w1.expression(ppf_expr, {'w': w1}).multiply(-1))
    output = output0.where(mask1, output1)
    return output


def date_range(start_dt, end_dt, days=1, skip_leap_days=True):
    """Generate dates within a range (inclusive)"""
    curr_dt = start_dt
    while curr_dt <= end_dt:
        if not skip_leap_days or curr_dt.month != 2 or curr_dt.day != 29:
            yield curr_dt
        curr_dt += datetime.timedelta(days=days)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Earth Engine EDDI Image Download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', type=utils.arg_valid_file,
        help='Input file', metavar='FILE')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    args = parser.parse_args()

    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    else:
        args.ini = utils.get_ini_path(os.getcwd())
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.getLogger('googleapiclient').setLevel(logging.ERROR)
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format(
        'Start Time:', datetime.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    ee_image_download(ini_path=args.ini, overwrite_flag=args.overwrite)
