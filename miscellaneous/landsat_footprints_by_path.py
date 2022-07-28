import argparse
import datetime
import logging
import os
import sys

import ee


def main():
    """Generate custom WRS2 descending CONUS footprints"""
    logging.info('\nGenerate custom WRS2 descending CONUS footprints')

    gdrive_workspace = 'C:\Users\mortonc\Google Drive'
    export_folder = 'EE_Exports'
    # output_name = 'wrs2_descending_conus_custom'
    output_fmt = 'wrs2_descending_p{:03d}_custom'
    wrs2_ft = 'ft:1yZ9Q0gJL9t9NULWgx7RrLC7mWomL-OzF-7IuyCiL'

    scale = 120
    crs = 'EPSG:4326'
    min_count = 4
    max_pixels = 1E10

    # l5_skip_dates = [
    #     '1986-06-27', '1987-05-13', '1987-07-16', '1987-07-31', '1987-11-05',
    #     '1989-05-14', '1988-05-30', '1991-07-11', '1994-09-04', '1995-12-01',
    #     '1996-02-03', '1997-10-01', '2002-12-13', '2003-03-21', '2008-11-07'
    # ]
    # l7_skip_date = []
    # l8_skip_dates = [
    #     '2013-05-18', '2013-05-19', '2013-05-21', '2013-05-30']

    export_ws = os.path.join(gdrive_workspace, export_folder)
    if not os.path.isdir(export_ws):
        os.mkdir(export_ws)

    logging.info('  {}'.format(export_ws))

    ee.Initialize()

    # path_list = range(10, 49, 1)
    path_list = range(1, 47, 1)

    for path in path_list:
        logging.info('Path: {}'.format(path))
        logging.info('  {}'.format(output_fmt.format(path) + '.geojson'))

        pr_coll = ee.FeatureCollection(wrs2_ft) \
            .filterMetadata('PATH', 'equals', path)
        #     .filterMetadata('PATH_ROW', 'equals', 'p044r031')
        pr_geom = pr_coll.geometry()

        def pr_footprint_func(path_row):
            path = ee.Number(ee.Feature(path_row).get('PATH'))
            row = ee.Number(ee.Feature(path_row).get('ROW'))
            # path = ee.Number(ee.List(path_row).get(0));
            # row = ee.Number(ee.List(path_row).get(1));

            l5_coll = ee.ImageCollection('LANDSAT/LT5_L1T_TOA') \
                .filterBounds(pr_geom) \
                .filterMetadata('WRS_PATH', 'equals', path) \
                .filterMetadata('WRS_ROW', 'equals', row) \
                # .filter(ee.Filter.inList('DATE_ACQUIRED', l5_skip_dates).Not()) \
                .select(
                    ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'lst'])
            l7_coll = ee.ImageCollection('LANDSAT/LE7_L1T_TOA') \
                .filterBounds(pr_geom) \
                .filterMetadata('WRS_PATH', 'equals', path) \
                .filterMetadata('WRS_ROW', 'equals', row) \
                # .filter(ee.Filter.inList('DATE_ACQUIRED', l7_skip_dates).Not()) \
                .select(
                    ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6_VCID_1'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'lst'])
            l8_coll = ee.ImageCollection('LANDSAT/LC8_L1T_TOA') \
                .filterBounds(pr_geom) \
                .filterDate('2013-06-01', '2017-12-31') \
                .filterMetadata('WRS_PATH', 'equals', path) \
                .filterMetadata('WRS_ROW', 'equals', row) \
                # .filter(ee.Filter.inList('DATE_ACQUIRED', l8_skip_dates).Not()) \
                .select(
                    ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'lst'])

            landsat_coll = ee.ImageCollection([])
            landsat_coll = ee.ImageCollection(landsat_coll.merge(l5_coll))
            landsat_coll = ee.ImageCollection(landsat_coll.merge(l7_coll))
            landsat_coll = ee.ImageCollection(landsat_coll.merge(l8_coll))

            def common_area(image):
                return ee.Image(image).reduce(ee.Reducer.allNonZero())
            count_image = landsat_coll.map(common_area).sum()
            mask_image = ee.Image(count_image).gt(min_count)
            mask_image = mask_image.updateMask(mask_image)

            # landsat_image = ee.Image(landsat_coll.mosaic())
            # landsat_image = ee.ImageCollection([
            #   l5_coll.mosaic(), l7_coll.mosaic(), l8_coll.mosaic()]).mosaic();
            # common_image = ee.Image(landsat_image).mask().reduce(ee.Reducer.And())
            # mask_image = ee.Image(landsat_image).select([0]) \
            #     .multiply(0).add(1).int() \
            #     .updateMask(common_image)

            mask_bounds = landsat_coll.geometry(10).bounds(10).buffer(10000)

            mask_geom = mask_image \
                .reduceToVectors(
                    geometry=mask_bounds,
                    scale=scale,
                    crs=crs,
                    maxPixels=max_pixels) \
                .geometry() \
                .convexHull(10)

            return ee.Feature(mask_geom, {
                'PATH': path, 'ROW': row,
                'PATH_ROW': path_row.get('PATH_ROW'),
                # 'PATH_ROW': ee.String('p').cat(path.format('%03d')) \
                #     .cat('r').cat(row.format('%03d')),
                'EPSG': path_row.get('EPSG')
            })

        footprint_coll = ee.FeatureCollection(pr_coll.map(pr_footprint_func))

        logging.debug('  Building export task')
        task = ee.batch.Export.table.toDrive(
            collection=footprint_coll,
            description=output_fmt.format(path),
            folder=export_folder,
            # fileNamePrefix='',
            fileFormat='GeoJSON'
        )
        logging.debug('  Starting export task')
        try:
            task.start()
        except Exception as e:
            # logging.error('  Exception: {}, retry {}'.format(e, i))
            logging.error('{}'.format(e))
            # sleep(i ** 2)
        logging.info('  Active: {}'.format(task.active()))


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Generate custom WRS2 descending CONUS footprints',
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
