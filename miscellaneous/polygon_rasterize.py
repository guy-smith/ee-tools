import argparse
import logging
import math
import os
import subprocess

from osgeo import gdal, ogr, osr


def main(input_path, output_path, output_epsg=None,
         cellsize=30, snap=(15, 15), overwrite_flag=False):
    """Adjust shapefile polygon borders to follow raster cell outlines
      in an arbitrary projection and snap.

    Args:
        input_path (str):
        output_path (str):
        output_epsg (int): EPSG code
        cellsize (float): cellsize
        snap (list/tuple):
        overwrite_flag (bool): if True, overwrite existing files

    """
    logging.info('Rasterizing Polygon Geometry')
    logging.debug('  Input:  {}'.format(input_path))
    logging.debug('  Output: {}'.format(output_path))

    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    gdal_mem_driver = gdal.GetDriverByName('MEM')
    gdal_tif_driver = gdal.GetDriverByName('GTiff')
    ogr_mem_driver = ogr.GetDriverByName('MEMORY')

    if os.path.isfile(output_path):
        if not overwrite_flag:
            logging.warning(
                '  Output file already exists and overwrite=False, exiting')
            return False
        else:
            logging.debug('  Removing existing output file')
            shp_driver.DeleteDataSource(output_path)

    # Build the output spatial reference object
    if output_epsg is None:
        input_ds = ogr.Open(input_path, 0)
        input_lyr = input_ds.GetLayer()
        input_osr = input_lyr.GetSpatialRef()
        logging.debug('  Using input spatial reference: {}'.format(
            input_osr.ExportToProj4()))
        output_osr = input_osr.Clone()
        output_srs = input_osr.ExportToWkt()
        input_ds = None
        del input_ds, input_lyr
    else:
        output_osr = osr.SpatialReference()
        output_osr.ImportFromEPSG(int(output_epsg))
        output_srs = 'EPSG:{}'.format(output_epsg)

    # Use OGR utilities to project the shapefile
    subprocess.call([
        'ogr2ogr', '-overwrite', '-t_srs', output_srs,
        output_path, input_path])

    # Open the output shapefile and modify the geometry in place
    output_ds = ogr.Open(output_path, 1)
    output_lyr = output_ds.GetLayer()

    # Read the output layer into memory
    # This is a separate layer for filtering by FID and rasterizing
    memory_ds = ogr_mem_driver.CreateDataSource('memory')
    memory_lyr = memory_ds.CopyLayer(output_lyr, 'memory', ['OVERWRITE=YES'])

    # It may not be necessary to reset the layers
    memory_lyr.ResetReading()
    output_lyr.ResetReading()

    # Rasterize each feature separately
    for output_ftr in output_lyr:
        output_fid = output_ftr.GetFID()
        logging.debug('FID: {}'.format(output_fid))

        # Select the current feature from the input layer
        memory_lyr.SetAttributeFilter("{0} = {1}".format('FID', output_fid))
        input_ftr = memory_lyr.GetNextFeature()
        input_geom = input_ftr.GetGeometryRef()
        # logging.debug('  Geom: {}'.format(input_geom.ExportToWkt()))

        output_extent = ogrenv_swap(input_geom.GetEnvelope())
        # logging.debug('  Extent: {}'.format(output_extent))
        output_extent = adjust_to_snap(
            output_extent, 'EXPAND', snap[0], snap[1], cellsize)
        output_geo = extent_geo(output_extent, cellsize)
        logging.debug('  Extent: {}'.format(output_extent))
        logging.debug('  Geo:    {}'.format(output_geo))

        # Create the in-memory raster to rasterize into
        raster_rows, raster_cols = extent_shape(output_extent, cellsize)
        logging.debug('  Shape:  {}x{}'.format(raster_rows, raster_cols))
        raster_ds = gdal_mem_driver.Create(
            '', raster_cols, raster_rows, 1, gdal.GDT_Byte)
        # raster_ds = gdal_tif_driver.Create(
        #     'test.tif', raster_cols, raster_rows, 1, gdal.GDT_Byte)
        raster_ds.SetProjection(output_osr.ExportToWkt())
        raster_ds.SetGeoTransform(output_geo)
        raster_band = raster_ds.GetRasterBand(1)
        raster_band.Fill(0)
        raster_band.SetNoDataValue(0)

        # Rasterize the current feature
        gdal.RasterizeLayer(
            raster_ds, [1], memory_lyr, burn_values=[1])
        raster_band = raster_ds.GetRasterBand(1)

        # Polygonize the raster
        polygon_ds = ogr_mem_driver.CreateDataSource('memData')
        # polygon_ds = ogr_mem_driver.Open('memData', 1)
        polygon_lyr = polygon_ds.CreateLayer('memLayer', srs=None)
        gdal.Polygonize(
            raster_band, raster_band, polygon_lyr, -1, [], callback=None)
        raster_ds, raster_band = None, None

        # Get the new geometry from the in memory polygon
        output_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        # if polygon_lyr.GetFeatureCount() > 1:
        #     output_geom = ogr.Geometry(ogr.wkbMultiPolygon)
        # else:
        #     output_geom = ogr.Geometry(ogr.wkbPolygon)

        for polygon_ftr in polygon_lyr:
            output_geom.AddGeometry(polygon_ftr.GetGeometryRef())
        # logging.debug('  Geom: {}'.format(output_geom))

        # Replace the original geometry with the new geometry
        output_ftr.SetGeometry(output_geom)
        output_lyr.SetFeature(output_ftr)

        polygon_lyr, polygon_ds = None, None

    output_ds, output_lyr = None, None
    memory_ds, memory_lyr = None, None

    # Set the output spatial reference
    output_osr.MorphToESRI()
    with open(output_path.replace('.shp', '.prj'), 'w') as output_f:
        output_f.write(output_osr.ExportToWkt())


def adjust_to_snap(extent, method, snap_x, snap_y, cellsize):
    extent[0] = math.floor((extent[0] - snap_x) / cellsize) * cellsize + snap_x
    extent[1] = math.floor((extent[1] - snap_y) / cellsize) * cellsize + snap_y
    extent[2] = math.ceil((extent[2] - snap_x) / cellsize) * cellsize + snap_x
    extent[3] = math.ceil((extent[3] - snap_y) / cellsize) * cellsize + snap_y
    return extent


def ogrenv_swap(extent):
    return [extent[0], extent[2], extent[1], extent[3]]


def extent_shape(extent, cellsize):
    """Return number of rows and columns of the extent"""
    cols = int(round(abs((extent[0] - extent[2]) / cellsize), 0))
    rows = int(round(abs((extent[3] - extent[1]) / -cellsize), 0))
    return rows, cols


def extent_geo(extent, cellsize):
    return (extent[0], cellsize, 0., extent[3], 0., -cellsize)


def shp_type(param):
    if os.path.splitext(param)[1].lower() not in ['.shp']:
        raise argparse.ArgumentTypeError('File must have a ".shp" extension')
    return param


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Rasterize Polygon Geometry',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'src', type=shp_type, help='Input file path')
    parser.add_argument(
        'dst', help='Output file path')
    parser.add_argument(
        '-a_srs',
        help='Output spatial reference')
    parser.add_argument(
        '-e', '--epsg', default=None,
        help='Output spatial reference EPSG code')
    parser.add_argument(
        '-cs', '--cellsize', default=30, type=float,
        help='Output spatial reference')
    parser.add_argument(
        '-s', '--snap', default=[15, 15], type=float, nargs=2,
        help='Snap point (x, y)', metavar=('X', 'Y'))
    parser.add_argument(
        '-o', '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.src and os.path.isfile(os.path.abspath(args.src)):
        args.src = os.path.abspath(args.src)
    if args.dst and os.path.isfile(os.path.abspath(args.dst)):
        args.dst = os.path.abspath(args.dst)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    main(
        input_path=args.src, output_path=args.dst,
        output_epsg=args.epsg, cellsize=args.cellsize,
        snap=args.snap, overwrite_flag=args.overwrite)
