#--------------------------------
# Name:         gdal_common.py
# Purpose:      Common GDAL support functions
# Python:       3.6
#--------------------------------

import copy
import json
import logging
import math
import os
import sys

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()


class Extent:
    """Bounding Geographic Extent"""
    # def __repr__(self):
    #     return '<Extent xmin:{0} ymin:{1} xmax:{2} ymax:{3}>'.format(
    #         self.xmin, self.ymin, self.xmax, self.ymax)

    def __str__(self):
        return '{0} {1} {2} {3}'.format(
            self.xmin, self.ymin, self.xmax, self.ymax)

    def __iter__(self):
        return iter((self.xmin, self.ymin, self.xmax, self.ymax))

    def __init__(self, extent, ndigits=10):
        """Round values to avoid Float32 rounding errors"""
        (xmin, ymin, xmax, ymax) = extent
        self.xmin = round(xmin, ndigits)
        self.ymin = round(ymin, ndigits)
        self.xmax = round(xmax, ndigits)
        self.ymax = round(ymax, ndigits)

    def adjust_to_snap(self, method, snap_x, snap_y, cs):
        if method.upper() == 'ROUND':
            xmin = math.floor((self.xmin - snap_x) / cs + 0.5) * cs + snap_x
            ymin = math.floor((self.ymin - snap_y) / cs + 0.5) * cs + snap_y
            xmax = math.floor((self.xmax - snap_x) / cs + 0.5) * cs + snap_x
            ymax = math.floor((self.ymax - snap_y) / cs + 0.5) * cs + snap_y
        elif method.upper() == 'EXPAND':
            xmin = math.floor((self.xmin - snap_x) / cs) * cs + snap_x
            ymin = math.floor((self.ymin - snap_y) / cs) * cs + snap_y
            xmax = math.ceil((self.xmax - snap_x) / cs) * cs + snap_x
            ymax = math.ceil((self.ymax - snap_y) / cs) * cs + snap_y
        elif method.upper() == 'SHRINK':
            xmin = math.ceil((self.xmin - snap_x) / cs) * cs + snap_x
            ymin = math.ceil((self.ymin - snap_y) / cs) * cs + snap_y
            xmax = math.floor((self.xmax - snap_x) / cs) * cs + snap_x
            ymax = math.floor((self.ymax - snap_y) / cs) * cs + snap_y

        return Extent((xmin, ymin, xmax, ymax))

        # if method.upper() == 'ROUND':
        #     self.xmin = math.floor((self.xmin - snap_x) / cs + 0.5) * cs + snap_x
        #     self.ymin = math.floor((self.ymin - snap_y) / cs + 0.5) * cs + snap_y
        #     self.xmax = math.floor((self.xmax - snap_x) / cs + 0.5) * cs + snap_x
        #     self.ymax = math.floor((self.ymax - snap_y) / cs + 0.5) * cs + snap_y
        # elif method.upper() == 'EXPAND':
        #     self.xmin = math.floor((self.xmin - snap_x) / cs) * cs + snap_x
        #     self.ymin = math.floor((self.ymin - snap_y) / cs) * cs + snap_y
        #     self.xmax = math.ceil((self.xmax - snap_x) / cs) * cs + snap_x
        #     self.ymax = math.ceil((self.ymax - snap_y) / cs) * cs + snap_y
        # elif method.upper() == 'SHRINK':
        #     self.xmin = math.ceil((self.xmin - snap_x) / cs) * cs + snap_x
        #     self.ymin = math.ceil((self.ymin - snap_y) / cs) * cs + snap_y
        #     self.xmax = math.floor((self.xmax - snap_x) / cs) * cs + snap_x
        #     self.ymax = math.floor((self.ymax - snap_y) / cs) * cs + snap_y

    def buffer(self, distance):
        xmin = self.xmin - distance
        ymin = self.ymin - distance
        xmax = self.xmax + distance
        ymax = self.ymax + distance
        return Extent((xmin, ymin, xmax, ymax))
        # self.xmin -= distance
        # self.ymin -= distance
        # self.xmax += distance
        # self.ymax += distance

    # DEADBEEF
    # def split(self):
    #     """List of extent terms (xmin, ymin, xmax, ymax)

    #     This method is redundant to __iter__
    #     """
    #     return [self.xmin, self.ymin, self.xmax, self.ymax]

    def copy(self):
        """Return a copy of the extent"""
        return Extent((self.xmin, self.ymin, self.xmax, self.ymax))

    def corner_points(self):
        """Corner points in clockwise order starting with upper-left point"""
        return [(self.xmin, self.ymax), (self.xmax, self.ymax),
                (self.xmax, self.ymin), (self.xmin, self.ymin)]

    def ul_lr_swap(self):
        """Copy of extent object reordered as xmin, ymax, xmax, ymin

        Some gdal utilities want the extent described using upper-left and
        lower-right points.
            gdal_translate -projwin ulx uly lrx lry
            gdal_merge -ul_lr ulx uly lrx lry

        """
        return Extent((self.xmin, self.ymax, self.xmax, self.ymin))

    def ogrenv_swap(self):
        """Copy of extent object reordered as xmin, xmax, ymin, ymax

        OGR feature (shapefile) extents are different than GDAL raster extents
        """
        return Extent((self.xmin, self.xmax, self.ymin, self.ymax))

    def origin(self):
        """Origin (upper-left corner) of the extent"""
        return (self.xmin, self.ymax)

    def center(self):
        """Centroid of the extent"""
        return ((self.xmin + 0.5 * (self.xmax - self.xmin)),
                (self.ymin + 0.5 * (self.ymax - self.ymin)))

    def shape(self, cs):
        """Return number of rows and columns of the extent

        Args:
            cs (int): cellsize

        Returns:
            tuple of raster rows and columns
        """
        cols = int(round(abs((self.xmin - self.xmax) / cs), 0))
        rows = int(round(abs((self.ymax - self.ymin) / -cs), 0))
        return rows, cols

    def geo(self, cs):
        """Geo-tranform of the extent"""
        return (self.xmin, abs(cs), 0., self.ymax, 0., -abs(cs))

    def geometry(self):
        """GDAL geometry object of the extent"""
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in self.corner_points():
            ring.AddPoint(point[0], point[1])
        ring.CloseRings()
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        return polygon

    def intersect_point(self, xy):
        """"Test if Point XY intersects the extent"""
        if ((xy[0] > self.xmax) or
                (xy[0] < self.xmin) or
                (xy[1] > self.ymax) or
                (xy[1] < self.ymin)):
            return False
        else:
            return True


def raster_driver(raster_path):
    """Return the GDAL driver from a raster path

    Currently supports ERDAS Imagine format, GeoTiff,
    HDF-EOS (HDF4), BSQ/BIL/BIP, and memory drivers.

    Args:
        raster_path (str): filepath to a raster

    Returns:
        GDAL driver: GDAL raster driver

    """
    if raster_path.upper().endswith('IMG'):
        return gdal.GetDriverByName('HFA')
    elif raster_path.upper().endswith('TIF'):
        return gdal.GetDriverByName('GTiff')
    elif raster_path.upper().endswith('TIFF'):
        return gdal.GetDriverByName('GTiff')
    elif raster_path.upper().endswith('HDF'):
        return gdal.GetDriverByName('HDF4')
    elif raster_path[-3:].upper() in ['BSQ', 'BIL', 'BIP']:
        return gdal.GetDriverByName('EHdr')
    elif raster_path == '':
        return gdal.GetDriverByName('MEM')
    else:
        raise SystemExit('Unsupported or invalid raster extension: '
                         '{}'.format(os.path.basename(raster_path)))
        # raise ValueError('Unsupported or invalid raster extension: '
        #                  '{}'.format(os.path.basename(raster_path)))


def numpy_to_gdal_type(numpy_type):
    """Return the GDAL raster data type based on the NumPy array data type

    The following built in functions do roughly the same thing
        NumericTypeCodeToGDALTypeCode
        GDALTypeCodeToNumericTypeCode

    Args:
        numpy_type (:class:`np.dtype`): NumPy array type
            (i.e. np.bool, np.float32, etc)

    Returns:
        g_type: GDAL `datatype <http://www.gdal.org/gdal_8h.html#a22e22ce0a55036a96f652765793fb7a4/>`
        _equivalent to the input NumPy :class:`np.dtype`

    """
    if numpy_type == np.bool:
        return gdal.GDT_Byte
    elif numpy_type == np.int:
        return gdal.GDT_Int32
    elif numpy_type == np.int8:
        return gdal.GDT_Int16
    elif numpy_type == np.int16:
        return gdal.GDT_Int16
    elif numpy_type == np.int32:
        return gdal.GDT_Int32
    elif numpy_type == np.uint8:
        return gdal.GDT_Byte
    elif numpy_type == np.uint16:
        return gdal.GDT_UInt16
    elif numpy_type == np.uint32:
        return gdal.GDT_UInt32
    elif numpy_type == np.float:
        return gdal.GDT_Float64
    # elif numpy_type == np.float16:
    #     return gdal.GDT_Float32
    elif numpy_type == np.float32:
        return gdal.GDT_Float32
    elif numpy_type == np.float64:
        return gdal.GDT_Float32
    else:
        raise SystemExit('Unsupported or invalid NumPy array dtype: '
                         '{}'.format(numpy_type))
        # raise ValueError('Unsupported or invalid NumPy array dtype: '
        #                  '{}'.format(numpy_type))


def numpy_type_nodata(numpy_type):
    """Return the default nodata value based on the NumPy array data type

    Args:
        numpy_type (:class:`np.dtype`): numpy array type
            (i.e. np.bool, np.float32, etc)

    Returns:
        nodata_value: Nodata value for GDAL which defaults to the
            minimum value for the number type

    """
    if numpy_type == np.bool:
        return 0
    elif numpy_type == np.int:
        return int(np.iinfo(np.int32).min)
    elif numpy_type == np.int8:
        return int(np.iinfo(np.int8).min)
    elif numpy_type == np.int16:
        return int(np.iinfo(np.int16).min)
    elif numpy_type == np.int32:
        return int(np.iinfo(np.int32).min)
    elif numpy_type == np.uint8:
        return int(np.iinfo(np.uint8).max)
    elif numpy_type == np.uint16:
        return int(np.iinfo(np.uint16).max)
    elif numpy_type == np.uint32:
        return int(np.iinfo(np.uint32).max)
    elif numpy_type == np.float:
        return float(np.finfo(np.float64).min)
    elif numpy_type == np.float16:
        return float(np.finfo(np.float32).min)
    elif numpy_type == np.float32:
        return float(np.finfo(np.float32).min)
    elif numpy_type == np.float64:
        return float(np.finfo(np.float32).min)
    else:
        raise SystemExit('Unsupported or invalid NumPy array dtype: '
                         '{}'.format(numpy_type))
        # raise ValueError('Unsupported or invalid NumPy array dtype: '
        #                  '{}'.format(numpy_type))


def gdal_to_numpy_type(gdal_type):
    """Return the NumPy array data type based on a GDAL type

    Args:
        gdal_type (:class:`gdal.type`): GDAL data type

    Returns:
        numpy_type: NumPy datatype (:class:`np.dtype`)

    """
    if gdal_type == gdal.GDT_Unknown:
        return np.float64
    elif gdal_type == gdal.GDT_Byte:
        return np.uint8
    elif gdal_type == gdal.GDT_UInt16:
        return np.uint16
    elif gdal_type == gdal.GDT_Int16:
        return np.int16
    elif gdal_type == gdal.GDT_UInt32:
        return np.uint32
    elif gdal_type == gdal.GDT_Int32:
        return np.int32
    elif gdal_type == gdal.GDT_Float32:
        return np.float32
    elif gdal_type == gdal.GDT_Float64:
        return np.float64
    else:
        raise SystemExit('Unsupported or invalid GDAL data type: '
                         '{}'.format(gdal_type))
        # raise ValueError('Unsupported or invalid GDAL data type: '
        #                  '{}'.format(gdal_type))


def matching_spatref(osr_a, osr_b):
    """Test if two spatial reference objects match

    Compare common components of PROJ4 strings

    Args:
        osr_a: OSR spatial reference object
        osr_b: OSR spatial reference object

    Returns:
        Bool: True if OSR objects match. Otherwise, False.

    """
    proj4_a = str(osr_a.ExportToProj4()).split(' ')
    proj4_b = str(osr_b.ExportToProj4()).split(' ')
    proj4_a = dict([
        x.split('=') if '=' in x else [x, ''] for x in proj4_a if x])
    proj4_b = dict([
        x.split('=') if '=' in x else [x, ''] for x in proj4_b if x])

    common = set(proj4_a.keys()) & set(proj4_b.keys())
    if (sorted([v for k, v in proj4_a.items() if k in common]) ==
            sorted([v for k, v in proj4_b.items() if k in common])):
        return True
    else:
        return False


# def matching_spatref(osr_a, osr_b):
#     """Test if two spatial reference objects match

#     Args:
#         osr_a: OSR spatial reference object
#         osr_b: OSR spatial reference object

#     Returns:
#         Bool: True if OSR objects match. Otherwise, False.

#     """
#     osr_a.MorphToESRI()
#     osr_b.MorphToESRI()
#     if ((osr_a.GetAttrValue('GEOGCS') == osr_b.GetAttrValue('GEOGCS')) and
#           (osr_a.GetAttrValue('PROJECTION') == osr_b.GetAttrValue('PROJECTION')) and
#           (osr_a.GetAttrValue('UNIT') == osr_b.GetAttrValue('UNIT')) and
#           (osr_a.GetUTMZone() == osr_b.GetUTMZone())):
#         return True
#     else:
#         return False


def epsg_osr(input_epsg):
    """Return the spatial reference object of an EPSG code

    Args:
        input_epsg (int): EPSG spatial reference code as integer

    Returns:
        osr.SpatialReference: :class:`osr.SpatialReference` object

    """
    input_osr = osr.SpatialReference()
    result = input_osr.ImportFromEPSG(input_epsg)
    if result == 0:
        return input_osr
    else:
        raise SystemExit('Could not set spatial reference from EPSG:'
                         ' {}'.format(input_epsg))
        # raise ValueError('Could not set spatial reference from EPSG:'
        #                  ' {}'.format(input_epsg))


def wkt_osr(input_wkt):
    """Return the spatial reference object of a projection WKT
    Args:
        input_proj (:class:`osr.SpatialReference` WKT): Input
            WKT formatted :class:`osr.SpatialReference` object
            to be used in creation of an :class:`osr.SpatialReference`
    Returns:
        osr.SpatialReference: OSR SpatialReference object as represented
            by the input WKT
    """
    input_osr = osr.SpatialReference()
    result = input_osr.ImportFromWkt(input_wkt)
    if result == 0:
        return input_osr
    else:
        raise SystemExit('Could not set spatial reference from WKT:'
                         ' {}'.format(input_wkt))
        # raise ValueError('Could not set spatial reference from WKT:'
        #                  ' {}'.format(input_epsg))


def osr_wkt(input_osr):
    """Return the projection WKT of a spatial reference object

    Args:
        input_osr (:class:`osr.SpatialReference`): the input OSR
            spatial reference

    Returns:
        WKT: :class:`osr.SpatialReference` in WKT format

    """
    return input_osr.ExportToWkt()


def proj4_osr(input_proj4):
    """Return the spatial reference object of a PROJ4 string

    Args:
        input_proj4 (str): PROJ4 projection or coordinate system description

    Returns:
        osr.SpatialReference: :class:`osr.SpatialReference` object

    """
    input_osr = osr.SpatialReference()
    result = input_osr.ImportFromProj4(input_proj4)
    if result == 0:
        return input_osr
    else:
        raise SystemExit('Could not set spatial reference from PROJ4:\n'
                         '  {}'.format(input_proj4))
        # raise ValueError('Could not set spatial reference from PROJ4:\n'
        #                  '  {}'.format(input_proj4))


def osr_proj4(input_osr):
    """Return the PROJ4 code of an osr.SpatialReference
    Args:
        input_osr (:class:`osr.SpatialReference`): OSR Spatial reference
            of the input projection/GCS
    Returns:
        str: Proj4 string of the projection or GCS
    """
    return input_osr.ExportToProj4()


def feature_path_osr(feature_path):
    """Return the spatial reference of a feature path

    Args:
        feature_path (str): file path to the OGR supported feature

    Returns:
        osr.SpatialReference: :class:`osr.SpatialReference` of the
            input feature file path

    """
    feature_ds = ogr.Open(feature_path)
    feature_osr = feature_ds_osr(feature_ds)
    feature_ds = None
    return feature_osr


def feature_ds_osr(feature_ds):
    """Return the spatial reference of an opened feature dataset

    Args:
        feature_ds (:class:`ogr.Datset`): Opened feature dataset
            from which you desire the spatial reference

    Returns:
        osr.SpatialReference: :class:`osr.SpatialReference` of the input
            OGR feature dataset

    """
    feature_lyr = feature_ds.GetLayer()
    return feature_lyr_osr(feature_lyr)


def feature_lyr_osr(feature_lyr):
    """Return the spatial reference of a feature layer

    Args:
        feature_lyr (:class:`ogr.Layer`): OGR feature layer from
            which you desire the :class:`osr.SpatialReference`

    Returns:
        osr.SpatialReference: the :class:`osr.SpatialReference` object
            of the input feature layer

    """
    return feature_lyr.GetSpatialRef()


def feature_lyr_extent(feature_lyr):
    """Return the extent of an opened feature layer

    Args:
        feature_lyr (:class:`ogr.Layer`): An OGR feature
            layer

    Returns:
        gdal_common.extent: :class:`gdal_common.extent` of the
            input feature layer

    """
    # OGR Extent format (xmin, xmax, ymin, ymax)
    # ArcGIS/GDAL(?) Extent format (xmin, ymin, xmax, ymax)
    f_extent = Extent(feature_lyr.GetExtent())
    f_env = f_extent.ogrenv_swap()
    # f_extent.ymin, f_extent.xmax = f_extent.xmax, f_extent.ymin
    return f_env


def raster_ds_geo(raster_ds, ndigits=None):
    """Return the geo-transform of an opened raster dataset

    Args:
        raster_ds (:class:`gdal.Dataset`): An Opened gdal raster dataset

    Returns:
        tuple: :class:`gdal.Geotransform` of the input dataset

    """
    if ndigits is not None:
        return round_geo(raster_ds.GetGeoTransform(), ndigits)
    else:
        return raster_ds.GetGeoTransform()


def round_geo(geo, ndigits):
    """Round the values of a geotransform to n digits

    Args:
        geo (tuple): :class:`gdal.Geotransform` object
        n (int): number of digits to round the
            :class:`gdal.Geotransform` to

    Returns:
        tuple: :class:`gdal.Geotransform` rounded to n digits

    """
    return tuple([round(i, ndigits) for i in geo])


def raster_ds_extent(raster_ds):
    """Return the extent of an opened raster dataset

    Args:
        raster_ds (:class:`gdal.Dataset`): An opened GDAL raster
            dataset

    Returns:
        tuple: :class:`gdal_common.extent` of the input dataset

    """
    raster_rows, raster_cols = raster_ds_shape(raster_ds)
    raster_geo = raster_ds_geo(raster_ds)
    return geo_extent(raster_geo, raster_rows, raster_cols)


def geo_cellsize(geo, x_only=False):
    """Return pixel width & pixel height of geo-transform

    Args:
        geo (tuple): :class:`gdal.Geotransform` object
        x_only (bool): If True, only return cell width

    Returns:
        tuple: tuple containing the x or x and y cellsize
    """
    if x_only:
        return geo[1]
    else:
        return (geo[1], geo[5])


def geo_origin(geo):
    """Return upper-left corner of geo-transform

    Returns the upper-left corner cordinates of :class:`GDAL.Geotransform`
    with the coordinates returned in the same projection/GCS as the input
    geotransform.

    Args:
        geo (:class:`GDAL.Geotransform`): Input GDAL Geotransform

    Returns:
        tuple:
        raster_origin: (x, y) coordinates of the upper left corner

    """
    return (geo[0], geo[3])


def geo_extent(geo, rows, cols):
    """Return the extent from a geo-transform and array shape

    This function takes the :class:`GDAL.Geotransform`, number of
    rows, and number of columns in a 2-dimensional :class:`np.array`
    (the :class:`np.array.shape`),and returns a :class:`gdc.extent`

    Geo-transform can be UL with +/- cellsizes or LL with +/+ cellsizes
    This approach should also handle UR and RR geo-transforms

    Returns ArcGIS/GDAL Extent format (xmin, ymin, xmax, ymax) but
        OGR Extent format (xmin, xmax, ymax, ymin) can be obtained using the
        extent.ul_lr_swap() method

    Args:
        geo (tuple): :class:`gdal.Geotransform` object
        rows (int): number of rows
        cols (int): number of cols

    Returns:
        gdal_common.extent:
        A :class:`gdal_common.extent` class object

    """
    cs_x, cs_y = geo_cellsize(geo, x_only=False)
    origin_x, origin_y = geo_origin(geo)
    # ArcGIS/GDAL Extent format (xmin, ymin, xmax, ymax)
    return Extent([min([origin_x + cols * cs_x, origin_x]),
                   min([origin_y + rows * cs_y, origin_y]),
                   max([origin_x + cols * cs_x, origin_x]),
                   max([origin_y + rows * cs_y, origin_y])])
    # OGR Extent format (xmin, xmax, ymax, ymin)
    # return Extent([origin_x, (origin_x + cols * cellsize),
    #                origin_y, (origin_y + rows * (-cellsize))])


def raster_path_shape(raster_path):
    """Return the number of rows and columns in a raster

    Args:
        raster_path (str): file path of the raster


    Returns:
        tuple of raster rows and columns
    """
    raster_ds = gdal.Open(raster_path, 0)
    raster_shape = raster_ds_shape(raster_ds)
    raster_ds = None
    return raster_shape


def raster_ds_shape(raster_ds):
    """Return the number of rows and columns in an opened raster dataset

    Args:
        raster_ds: opened raster dataset

    Returns:
        tuple of raster rows and columns
    """
    return raster_ds.RasterYSize, raster_ds.RasterXSize


def raster_path_set_nodata(raster_path, input_nodata):
    """Set raster nodata value for all bands"""
    raster_ds = gdal.Open(raster_path, 1)
    raster_ds_set_nodata(raster_ds, input_nodata)
    del raster_ds


def raster_ds_set_nodata(raster_ds, input_nodata):
    """Set raster dataset nodata value for all bands"""
    band_cnt = raster_ds.RasterCount
    for band_i in range(band_cnt):
        band = raster_ds.GetRasterBand(band_i + 1)
        band.SetNoDataValue(input_nodata)


def extents_overlap(a_extent, b_extent):
    """Test if two extents overlap"""
    if ((a_extent.xmin > b_extent.xmax) or
            (a_extent.xmax < b_extent.xmin) or
            (a_extent.ymin > b_extent.ymax) or
            (a_extent.ymax < b_extent.ymin)):
        return False
    else:
        return True


def union_extents(extent_list):
    """Return the union of all input extents"""
    common_extent = ()
    for image_extent in extent_list:
        if not common_extent:
            common_extent = copy.copy(image_extent)
        common_extent = Extent(
            (min(common_extent.xmin, image_extent.xmin),
             min(common_extent.ymin, image_extent.ymin),
             max(common_extent.xmax, image_extent.xmax),
             max(common_extent.ymax, image_extent.ymax)))
    return common_extent


def intersect_extents(extent_list):
    """Return the intersection of all input extents"""
    common_extent = ()
    for image_extent in extent_list:
        if not common_extent:
            common_extent = copy.copy(image_extent)
        common_extent = Extent(
            (max(common_extent.xmin, image_extent.xmin),
             max(common_extent.ymin, image_extent.ymin),
             min(common_extent.xmax, image_extent.xmax),
             min(common_extent.ymax, image_extent.ymax)))
    return common_extent


def project_extent(input_extent, input_osr, output_osr, cellsize):
    """Project extent to different spatial reference / coordinate system

    Args:
        input_extent (): the input gdal_common.extent to be reprojected
        input_osr (): OSR spatial reference of the input extent
        output_osr (): OSR spatial reference of the desired output
        cellsize (): the cellsize used to calculate the new extent.
            This cellsize is in the input spatial reference

    Returns:
        tuple: :class:`gdal_common.extent` in the desired projection
    """
    # Build an in memory feature to project to
    mem_driver = ogr.GetDriverByName('Memory')
    output_ds = mem_driver.CreateDataSource('')
    output_lyr = output_ds.CreateLayer(
        'projected_extent', geom_type=ogr.wkbPolygon)
    feature_defn = output_lyr.GetLayerDefn()
    # Place points at every "cell" between pairs of corner points
    ring = ogr.Geometry(ogr.wkbLinearRing)
    corners = input_extent.corner_points()
    for point_a, point_b in zip(corners, corners[1:]+[corners[0]]):
        if cellsize is None:
            steps = 1000
        else:
            steps = float(max(
                abs(point_b[0] - point_a[0]),
                abs(point_b[1] - point_a[1]))) / cellsize
        # steps = float(abs(point_b[0] - point_a[0])) / cellsize
        for x, y in zip(np.linspace(point_a[0], point_b[0], steps + 1),
                        np.linspace(point_a[1], point_b[1], steps + 1)):
            ring.AddPoint(x, y)
    ring.CloseRings()
    # Set the ring geometry into a polygon
    polygon = ogr.Geometry(ogr.wkbPolygon)
    polygon.AddGeometry(ring)
    # Project the geometry
    tx = osr.CoordinateTransformation(input_osr, output_osr)
    polygon.Transform(tx)
    # Create a new feature and set the geometry into it
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(polygon)
    # Add the feature to the output layer
    output_lyr.CreateFeature(feature)
    # Get the extent from the projected polygon
    return feature_lyr_extent(output_lyr)


def array_offset_geo(full_geo, x_offset, y_offset):
    """Return sub_geo that is offset from full_geo

    Args:
        full_geo (): gdal.geotransform to create the offset geotransform
        x_offset (): number of cells to move in x direction
        y_offset (): number of cells to move in y direction

    Returns:
        gdal.Geotransform offset by the spefiied number of x/y cells
    """
    sub_geo = list(full_geo)
    sub_geo[0] += x_offset * sub_geo[1]
    sub_geo[3] += y_offset * sub_geo[5]
    return tuple(sub_geo)


def array_geo_offsets(full_geo, sub_geo, cs):
    """Return x/y offset of a gdal.geotransform based on another gdal.geotransform

    Args:
        full_geo (): larger gdal.geotransform from which the offsets should be calculated
        sub_geo (): smaller form
        cs (int): cellsize

    Returns:
        x_offset: number of cells of the offset in the x direction
        y_offset: number of cells of the offset in the y direction
    """
    # Return UPPER LEFT array coordinates of sub_geo in full_geo
    # If portion of sub_geo is outside full_geo, only return interior portion
    x_offset = int(round((sub_geo[0] - full_geo[0]) / cs, 0))
    y_offset = int(round((sub_geo[3] - full_geo[3]) / -cs, 0))
    # Force offsets to be greater than zero
    x_offset, y_offset = max(x_offset, 0), max(y_offset, 0)
    return x_offset, y_offset


def raster_to_array(input_raster, band=1, mask_extent=None,
                    default_nodata_value=None, return_nodata=True):
    """Return a NumPy array from a raster

    Output array size will match the mask_extent if mask_extent is set

    Args:
        input_raster (str): Filepath to the raster for array creation
        band (int): band to convert to array in the input raster
        mask_extent: Mask defining desired portion of raster
        default_nodata_value (float): Value to use if not set in raster.
            The raster nodata or this value will be used to initialize
                the output arrays.
        return_nodata (bool): If True, return no data value with the array

    Returns:
        output_array: The array of the raster values
        output_nodata: No data value of the raster file
    """
    input_raster_ds = gdal.Open(input_raster, 0)
    output_array, output_nodata = raster_ds_to_array(
        input_raster_ds, band, mask_extent, default_nodata_value,
        return_nodata=True)
    input_raster_ds = None
    if return_nodata:
        return output_array, output_nodata
    else:
        return output_array


def raster_ds_to_array(input_raster_ds, band=1, mask_extent=None,
                       default_nodata_value=None, return_nodata=True):
    """Return a NumPy array from an opened raster dataset

    Output array size will match the mask_extent if mask_extent is set

    Args:
        input_raster_ds (): opened raster dataset as gdal raster
        band (int): band number to read the array from
        mask_extent (): subset extent of the raster if desired
        default_nodata_value (float): Value to use if not set in raster.
            The raster nodata value or this value will be used to initialize
                the output arrays.
        return_nodata (bool): If True, return no data value with the array

    Returns:
        output_array: The array of the raster values
        output_nodata: No data value of the raster file
    """
    # DEADBEEF - User should be able to pass in an output nodata value
    input_extent = raster_ds_extent(input_raster_ds)
    input_geo = raster_ds_geo(input_raster_ds)
    input_cs = geo_cellsize(input_geo, x_only=True)
    input_rows, input_cols = raster_ds_shape(input_raster_ds)
    input_band = input_raster_ds.GetRasterBand(band)
    input_type = input_band.DataType
    numpy_type = gdal_to_numpy_type(input_type)
    input_nodata = input_band.GetNoDataValue()
    # Use default_nodata_value as the raster nodata value if raster doesn't have a
    #   nodata value set
    if input_nodata is None and default_nodata_value is not None:
        input_nodata = default_nodata_value
    # If raster doesn't have a nodata value and default_nodata_value isn't set
    #   use default nodata value for raster data type
    elif input_nodata is None and default_nodata_value is None:
        input_nodata = numpy_type_nodata(numpy_type)

    #
    if mask_extent:
        mask_rows, mask_cols = mask_extent.shape(input_cs)
        # If extents don't overlap, array is all nodata
        if not extents_overlap(input_extent, mask_extent):
            output_array = np.zeros((mask_rows, mask_cols), dtype=numpy_type)
            output_array[:] = input_nodata
        # Get intersecting portion of input array
        else:
            mask_geo = mask_extent.geo(input_cs)
            int_extent = intersect_extents([input_extent, mask_extent])
            int_geo = int_extent.geo(input_cs)
            int_xi, int_yi = array_geo_offsets(input_geo, int_geo, input_cs)
            int_rows, int_cols = int_extent.shape(input_cs)
            output_array = np.empty((mask_rows, mask_cols), dtype=numpy_type)
            output_array[:] = input_nodata
            m_xi, m_yi = array_geo_offsets(mask_geo, int_geo, input_cs)
            m_xf = m_xi + int_cols
            m_yf = m_yi + int_rows
            output_array[m_yi:m_yf, m_xi:m_xf] = input_band.ReadAsArray(
                int_xi, int_yi, int_cols, int_rows)
    else:
        output_array = input_band.ReadAsArray(
            0, 0, input_raster_ds.RasterXSize, input_raster_ds.RasterYSize)
    # For float types, set nodata values to nan
    if output_array.dtype in [np.float32, np.float64]:
        output_nodata = np.nan
        if input_nodata is not None:
            output_array[output_array == input_nodata] = output_nodata
    else:
        output_nodata = input_nodata
    if return_nodata:
        return output_array, output_nodata
    else:
        return output_array


def project_array(input_array, resampling_type,
                  input_osr, input_cs, input_extent,
                  output_osr, output_cs, output_extent,
                  output_nodata=None):
    """Project a NumPy array to a new spatial reference

    This function doesn't correctly handle masked arrays
    Must pass output_extent & output_cs to get output raster shape
    There is not enough information with just output_geo and output_cs

    Args:
        input_array (array: :class:`numpy.array`):
        resampling_type ():
        input_osr (:class:`osr.SpatialReference):
        input_cs (int):
        input_extent ():
        output_osr (:class:`osr.SpatialReference):
        output_cs (int):
        output_extent ():
        output_nodata (float):

    Returns:
        array: :class:`numpy.array`
    """

    # If input array has 3 dimensions, assume 1st dimension is time
    input_shape = input_array.shape
    input_dims = len(input_array.shape)
    if input_dims == 3:
        band_cnt, input_rows, input_cols = input_shape
    elif input_dims == 2:
        band_cnt = 1
        input_rows, input_cols = input_shape
    else:
        logging.error('Project array can not currently handle an ' +
                      'input array with shape {}'.format(input_shape))
        sys.exit()

    input_gtype = numpy_to_gdal_type(input_array.dtype)
    input_nodata = numpy_type_nodata(input_array.dtype)

    # If input array has nan, make a copy in order to set nodata values
    copy_array = np.array(input_array, copy=True)
    if (input_array.dtype in [np.float32, np.float64] and
            np.isnan(copy_array).any()):
        copy_array[np.isnan(copy_array)] = input_nodata

    # For 2d arrays, insert an a "band" dimension at the beginning
    if input_dims == 2:
        copy_array = np.expand_dims(copy_array, axis=0)

    # Create an in memory raster to store the array
    # ReprojectImage only works on raster datasets, not arrays
    mem_driver = gdal.GetDriverByName('MEM')
    input_ds = mem_driver.Create(
        '', input_cols, input_rows, band_cnt, input_gtype)
    input_ds.SetProjection(osr_proj(input_osr))
    input_ds.SetGeoTransform(input_extent.geo(input_cs))
    for band_i in xrange(band_cnt):
        input_band = input_ds.GetRasterBand(band_i + 1)
        input_band.SetNoDataValue(input_nodata)
        input_band.WriteArray(copy_array[band_i, :, :], 0, 0)
    del copy_array

    # Build the output raster to store the projected array
    output_rows, output_cols = output_extent.shape(output_cs)
    output_ds = mem_driver.Create(
        '', output_cols, output_rows, band_cnt, input_gtype)
    output_ds.SetProjection(output_osr.ExportToWkt())
    output_ds.SetGeoTransform(output_extent.geo(output_cs))
    for band_i in xrange(band_cnt):
        output_band = output_ds.GetRasterBand(band_i + 1)
        output_band.SetNoDataValue(input_nodata)
        output_band.Fill(input_nodata)

    # Project the array to the output raster
    gdal.ReprojectImage(
        input_ds, output_ds, input_osr.ExportToWkt(),
        output_osr.ExportToWkt(), resampling_type)
    input_ds = None

    # Get the projected array from the output raster dataset
    output_array = np.full(
        (band_cnt, output_rows, output_cols),
        input_nodata, input_array.dtype)
    # output_array = np.empty(
    #     (band_cnt, output_rows, output_cols), np.float32)
    for band_i in xrange(band_cnt):
        output_band = output_ds.GetRasterBand(band_i + 1)
        output_array[band_i, :, :] = output_band.ReadAsArray(
            0, 0, output_cols, output_rows)

    # For float types, set nodata values to nan
    if output_array.dtype in [np.float32, np.float64]:
        output_nodata = np.nan
        output_array[output_array == input_nodata] = output_nodata
    else:
        output_nodata = int(input_nodata)
    output_ds = None

    # Squeeze 3D back to 2D if necessary
    if input_dims == 3:
        return output_array
    if input_dims == 2:
        return np.squeeze(output_array, axis=0)


def shapefile_2_geom_list_func(input_path, zone_field=None,
                               reverse_flag=False, simplify_flag=False):
    """Return a list of feature geometries in the shapefile

    Also return the FID and value in zone_field
    """
    logging.info('\nReading zone shapefile')
    ftr_geom_list = []
    input_ds = ogr.Open(input_path)
    input_lyr = input_ds.GetLayer()
    input_ftr_defn = input_lyr.GetLayerDefn()
    # input_proj = input_lyr.GetSpatialRef().ExportToWkt()
    if not zone_field or zone_field.upper() == 'FID':
        zone_field_i = None
        logging.info('  Using FID as zone field')
    elif zone_field in feature_lyr_fields(input_lyr):
        zone_field_i = input_ftr_defn.GetFieldIndex(zone_field)
        logging.debug('  Zone field: {}  (index: {})'.format(
            zone_field, zone_field_i))
    else:
        logging.error('\nERROR: Zone field "{}" is not in the '
                      'shapefile'.format(zone_field))
        return []
        # raise ?

    input_ftr = input_lyr.GetNextFeature()
    while input_ftr:
        input_fid = input_ftr.GetFID()
        if zone_field_i is not None:
            input_zone = str(input_ftr.GetField(zone_field_i))
        else:
            input_zone = 'fid_{}'.format(input_fid)
        input_geom = input_ftr.GetGeometryRef()
        if simplify_flag:
            input_geom = input_geom.Simplify(1)
            reverse_flag = False

        # Convert feature to GeoJSON
        json_str = input_ftr.ExportToJson()
        json_obj = json.loads(json_str)

        # Reverse the point order from counter-clockwise to clockwise
        if reverse_flag:
            json_obj['geometry'] = json_reverse_func(json_obj['geometry'])

        # Strip Z value from coordinates
        json_obj['geometry'] = json_strip_z_func(json_obj['geometry'])
        # DEADBEEF - This check is happening inside the function and
        #   is probably redundant here.
        # if ((json_obj['geometry']['type'].lower() in ['polygon', 'multipolygon']) and
        #         (len(json_obj['geometry']['coordinates'][0][0]) == 3)):

        # # Reverse the point order from counter-clockwise to clockwise
        # if reverse_flag and input_geom.GetGeometryName() == 'MULTIPOLYGON':
        #     for i in range(len(json_obj['geometry']['coordinates'])):
        #         for j in range(len(json_obj['geometry']['coordinates'][i])):
        #             json_obj['geometry']['coordinates'][i][j] = list(reversed(
        #                 json_obj['geometry']['coordinates'][i][j]))
        # elif reverse_flag and input_geom.GetGeometryName() == 'POLYGON':
        #     for i in range(len(json_obj['geometry']['coordinates'])):
        #         json_obj['geometry']['coordinates'][i] = list(reversed(
        #             json_obj['geometry']['coordinates'][i]))

        # # Strip Z value from coordinates
        # elif (input_geom.GetGeometryName() == 'MULTIPOLYGON' and
        #         (len(json_obj['geometry']['coordinates'][0][0]) == 3)):
        #     for i in range(len(json_obj['geometry']['coordinates'])):
        #         for j in range(len(json_obj['geometry']['coordinates'][i])):
        #             json_obj['geometry']['coordinates'][i][j] = [
        #                 x[:2] for x in json_obj['geometry']['coordinates'][i][j]]
        # elif (input_geom.GetGeometryName() == 'POLYGON' and
        #         (len(json_obj['geometry']['coordinates'][0][0]) == 3)):
        #     for i in range(len(json_obj['geometry']['coordinates'])):
        #         json_obj['geometry']['coordinates'][i] = [
        #             x[:2] for x in json_obj['geometry']['coordinates'][i]]

        # Save the JSON object in the list
        ftr_geom_list.append([input_fid, input_zone, json_obj['geometry']])
        # ftr_geom_list.append({
        #     'fid': input_fid, 'name': input_zone,
        #     'json': json_obj['geometry']})
        input_geom = None
        input_ftr = input_lyr.GetNextFeature()
    input_ds = None
    return sorted(ftr_geom_list)


def feature_path_fields(feature_path):
    """"""
    feature_ds = ogr.Open(feature_path)
    field_list = feature_ds_fields(feature_ds)
    feature_ds = None
    return field_list


def feature_ds_fields(feature_ds):
    """"""
    feature_lyr = feature_ds.GetLayer()
    return feature_lyr_fields(feature_lyr)


def feature_lyr_fields(feature_lyr):
    """"""
    feature_lyr_defn = feature_lyr.GetLayerDefn()
    return [
        feature_lyr_defn.GetFieldDefn(i).GetNameRef()
        for i in range(feature_lyr_defn.GetFieldCount())]


def json_reverse_func(json_geom):
    """Reverse the point order from counter-clockwise to clockwise

    json_geom is modified in place

    Args:
        json_geom (dict): The geometry sub dictionary of a geojson.

    Returns:
        dict
    """
    if json_geom['type'].lower() == 'multipolygon':
        for i in range(len(json_geom['coordinates'])):
            for j in range(len(json_geom['coordinates'][i])):
                json_geom['coordinates'][i][j] = list(reversed(
                    json_geom['coordinates'][i][j]))
    elif json_geom['type'].lower() == 'polygon':
        for i in range(len(json_geom['coordinates'])):
            json_geom['coordinates'][i] = list(reversed(
                json_geom['coordinates'][i]))
    return json_geom


def json_strip_z_func(json_geom):
    """Strip Z value from coordinates

    json_geom is modified in place

    Args:
        json_geom (dict): The geometry sub dictionary of a geojson.

    Returns:
        dict
    """
    if (json_geom['type'].lower() == 'multipolygon' and
            (len(json_geom['coordinates'][0][0][0]) == 3)):
        for i in range(len(json_geom['coordinates'])):
            for j in range(len(json_geom['coordinates'][i])):
                json_geom['coordinates'][i][j] = [
                    x[:2] for x in json_geom['coordinates'][i][j]]
    elif (json_geom['type'].lower() == 'polygon' and
            (len(json_geom['coordinates'][0][0]) == 3)):
        for i in range(len(json_geom['coordinates'])):
            json_geom['coordinates'][i] = [
                x[:2] for x in json_geom['coordinates'][i]]
    return json_geom


def geo_2_ee_transform(gdal_geo):
    """Convert GDAL GeoTransform to EE style crsTransform

    EE: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
    GDAL: [xTranslation, xScale, xShearing, yTranslation, yShearing, yScale]
    """
    return tuple([gdal_geo[i] for i in [1, 2, 0, 4, 5, 3]])


def ee_transform_2_geo(gdal_geo):
    """Convert EE style crsTransform to GDAL GeoTransform

    EE: [xScale, xShearing, xTranslation, yShearing, yScale, yTranslation]
    GDAL: [xTranslation, xScale, xShearing, yTranslation, yShearing, yScale]
    """
    return tuple([gdal_geo[i] for i in [2, 0, 1, 5, 3, 4]])
