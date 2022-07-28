import os

import numpy as np
from osgeo import gdal, ogr, osr
import pytest

import ee_tools.gdal_common as gdc


# Test parameters
grid_params = [
    {'extent': [2000, 1000, 2100, 1080], 'cellsize': 10, 'epsg': 32611},
    # GRIDMET grid
    {
        'extent': [-124.792996724447, 25.0418561299642, -67.0429967244499, 49.4168561299628],
        'cellsize': 0.04166666666666, 'epsg': 4326}
]


@pytest.fixture(scope='module', params=grid_params)
def grid(request):
    return Grid(**request.param)


class Grid(object):
    """Test grid object"""
    def __init__(self, extent, cellsize, epsg):
        # Intentionally not making extent an extent object
        # Modify extent list like it would be modify when initialized
        self.extent = [round(x, ndigits=10) for x in extent]
        # self.extent = extent
        self.cellsize = cellsize

        # Is it bad to have these being built the same as in gdal_common?
        # These could be computed using the Extent methods instead
        self.geo = (
            self.extent[0], self.cellsize, 0.,
            self.extent[3], 0., -self.cellsize)
        self.transform = (
            self.cellsize, 0, self.extent[0],
            0, -self.cellsize, self.extent[3])
        self.cols = int(round(abs(
            (self.extent[0] - self.extent[2]) / self.cellsize), 0))
        self.rows = int(round(abs(
            (self.extent[3] - self.extent[1]) / -self.cellsize), 0))
        self.shape = (self.rows, self.cols)
        self.snap_x = self.extent[0]
        self.snap_y = self.extent[3]
        self.origin = (self.extent[0], self.extent[3])
        self.center = (
            self.extent[0] + 0.5 * abs(self.extent[2] - self.extent[0]),
            self.extent[1] + 0.5 * abs(self.extent[3] - self.extent[1]))

        # Spatial Reference
        self.epsg = epsg
        self.osr = osr.SpatialReference()
        self.osr.ImportFromEPSG(self.epsg)
        self.proj4 = self.osr.ExportToProj4()
        self.wkt = self.osr.ExportToWkt()
        # self.osr = gdc.epsg_osr(epsg)
        # self.proj4 = gdc.osr_proj4(self.osr)
        # self.wkt = gdc.osr_wkt(self.osr)


class Feature:
    """Test Feature object"""
    def __init__(self, grid):
        # Copy properties from Grid
        # How could I get these values automatically (or inherit them)?
        self.extent = grid.extent
        self.osr = grid.osr
        # self.cellsize = grid.cellsize

        # Create the feature dataset
        mem_driver = ogr.GetDriverByName('Memory')
        self.ds = mem_driver.CreateDataSource('')
        self.lyr = self.ds.CreateLayer(
            'test_feature', self.osr, geom_type=ogr.wkbPolygon)

        # Add some fields
        self.lyr.CreateField(ogr.FieldDefn("Name", ogr.OFTString))
        self.lyr.CreateField(ogr.FieldDefn("Lat", ogr.OFTReal))
        self.lyr.CreateField(ogr.FieldDefn("Lon", ogr.OFTReal))
        feature_defn = self.lyr.GetLayerDefn()

        # Place points at every "cell" between pairs of corner points
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in gdc.Extent(grid.extent).corner_points():
            ring.AddPoint(x, y)
        # corners = gdc.Extent(grid.extent).corner_points()
        # for point_a, point_b in zip(corners, corners[1:] + [corners[0]]):
        #     if grid.cellsize is None:
        #         steps = 1000
        #     else:
        #         steps = float(max(
        #             abs(point_b[0] - point_a[0]),
        #             abs(point_b[1] - point_a[1]))) / grid.cellsize
        #     # steps = float(abs(point_b[0] - point_a[0])) / cellsize
        #     for x, y in zip(np.linspace(point_a[0], point_b[0], steps + 1),
        #                     np.linspace(point_a[1], point_b[1], steps + 1)):
        #         ring.AddPoint(x, y)
        ring.CloseRings()

        # Set the ring geometry into a polygon
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)

        # Create a new feature and set the geometry into it
        feature = ogr.Feature(feature_defn)
        feature.SetGeometry(polygon)

        # Add the feature to the output layer
        self.lyr.CreateFeature(feature)


class Raster(Grid):
    """Test Raster object"""
    def __init__(self, grid, filename='', dtype=np.float32,
                 v_min=0, v_max=10, v_nodata=10, r_nodata='DEFAULT'):
        """The default array is in_memory Float32 with some nodata"""

        # Copy properties from Grid
        # How could I get these values automatically (or inherit them)?
        self.extent = grid.extent
        self.geo = grid.geo
        self.shape = grid.shape
        self.cols = grid.cols
        self.rows = grid.rows
        self.wkt = grid.wkt

        # Set the nodata value using the default for the array type
        # If nodat is None, don't set the band nodata value
        if r_nodata == 'DEFAULT':
            self.nodata = gdc.numpy_type_nodata(dtype)
        elif r_nodata is None:
            self.nodata = None
        else:
            self.nodata = r_nodata

        # Array/Raster Type
        self.dtype = dtype
        # self.gtype = gdc.numpy_to_gdal_type(dtype)
        # self.gtype = gtype

        # Build the array as integers then cast to the output dtype
        self.array = np.random.randint(v_min, v_max + 1, size=self.shape)\
            .astype(self.dtype)
        # self.array = np.random.uniform(v_min, v_max, size=grid.shape)\
        #     .astype(self.dtype)

        # self.nodata = gdc.numpy_type_nodata(dtype)
        # self.nodata = nodata
        if self.dtype in [np.float32, np.float64] and v_nodata is not None:
            self.array[self.array == v_nodata] = np.nan

        # Filename can be an empty string
        self.path = filename
        driver = gdc.raster_driver(self.path)

        # Create the raster dataset
        self.gtype = gdc.numpy_to_gdal_type(dtype)
        self.ds = driver.Create(
            self.path, self.cols, self.rows, 1, self.gtype)
        self.ds.SetProjection(self.wkt)
        self.ds.SetGeoTransform(self.geo)

        # Write the array to the raster
        band = self.ds.GetRasterBand(1)
        if r_nodata is not None:
            band.SetNoDataValue(self.nodata)
        band.WriteArray(self.array, 0, 0)


# class GeoArray(Grid):
#     """Test GeoArray object"""
#     def __init__(self, grid, dtype, v_min, v_max, v_nodata):
#         # Copy properties from Grid
#         # How do I get these values automatically (or inherit them)?
#         self.extent = grid.extent
#         self.geo = grid.geo
#         self.shape = grid.shape
#         self.cols = grid.cols
#         self.rows = grid.rows
#         self.wkt = grid.wkt

#         # Array/Raster Type
#         self.dtype = dtype
#         # self.gtype = gdc.numpy_to_gdal_type(dtype)
#         # self.gtype = gtype

#         # Build the array as integers then cast to the output dtype
#         self.array = np.random.randint(v_min, v_max+1, size=grid.shape)\
#             .astype(self.dtype)
#         # self.array = np.random.uniform(v_min, v_max, size=grid.shape)\
#         #     .astype(self.dtype)

#         # self.nodata = gdc.numpy_type_nodata(dtype)
#         # self.nodata = nodata
#         if self.dtype in [np.float32, np.float64]:
#             self.array[self.array == v_nodata] = np.nan


class TestNumpy:
    def test_numpy_to_gdal_type(self):
        assert gdc.numpy_to_gdal_type(np.bool) == gdal.GDT_Byte
        assert gdc.numpy_to_gdal_type(np.uint8) == gdal.GDT_Byte
        assert gdc.numpy_to_gdal_type(np.float32) == gdal.GDT_Float32
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.numpy_to_gdal_type(None)

    def test_numpy_type_nodata(self):
        assert gdc.numpy_type_nodata(np.bool) == 0
        assert gdc.numpy_type_nodata(np.uint8) == 255
        assert gdc.numpy_type_nodata(np.int8) == -128
        assert gdc.numpy_type_nodata(np.float32) == float(
            np.finfo(np.float32).min)
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.numpy_type_nodata(None)


class TestGDAL:
    def test_raster_driver(self):
        assert gdc.raster_driver('')
        assert gdc.raster_driver('test.img')
        assert gdc.raster_driver('d:\\test\\test.tif')
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.raster_driver('test.abc')

    def test_gdal_to_numpy_type(self):
        assert gdc.gdal_to_numpy_type(gdal.GDT_Byte) == np.uint8
        assert gdc.gdal_to_numpy_type(gdal.GDT_Float32) == np.float32
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.gdal_to_numpy_type(None)


class TestOSR:
    """OSR specific tests"""
    def test_osr(self, epsg=32611):
        """Test building OSR from EPSG code to test GDAL_DATA environment variable

        This should fail if GDAL_DATA is not set or is invalid.
        """
        print('\nTesting GDAL_DATA environment variable')
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(epsg)
        wkt = sr.ExportToWkt()
        print('GDAL_DATA = {}'.format(os.environ['GDAL_DATA']))
        print('EPSG: {}'.format(epsg))
        print('WKT: {}'.format(wkt))
        assert wkt

        # This only exists the TestOSR class
        # raise pytest.UsageError(
        #     "GDAL_DATA environment variable not set, exiting")

    # Test the grid spatial reference parameters
    def test_osr_wkt(self, grid):
        """Return the projection WKT of a spatial reference object"""
        assert gdc.osr_wkt(grid.osr) == grid.wkt

    def test_osr_proj4(self, grid):
        """Return the projection PROJ4 string of a spatial reference object"""
        assert gdc.osr_proj4(grid.osr) == grid.proj4

    def test_epsg_osr(self, grid):
        """Return the spatial reference object of an EPSG code"""

        # Check that a bad EPSG code raises an exception
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.epsg_osr(-1)

        # Check that an OSR object is returned
        assert isinstance(
            gdc.epsg_osr(grid.epsg), type(osr.SpatialReference()))

    def test_proj4_osr(self, grid):
        """Return the spatial reference object of a PROj4 string"""
        # Check that a bad PROJ4 string raises an exception
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.proj4_osr('')

        # Check that an OSR object is returned
        assert isinstance(
            gdc.proj4_osr(grid.proj4), type(osr.SpatialReference()))

    def test_wkt_osr(self, grid):
        """Return the spatial reference object of a WKT string"""
        # Check that a bad WKT string raises an exception
        # with pytest.raises(ValueError):
        with pytest.raises(SystemExit):
            gdc.proj4_osr('')

        # Check that an OSR object is returned
        assert isinstance(
            gdc.wkt_osr(grid.wkt), type(osr.SpatialReference()))

    def test_matching_spatref(self, grid):
        """Compare the PROJ4 strings of two spatial reference objects"""
        a = grid.osr
        b = gdc.proj4_osr(grid.proj4)
        c = gdc.wkt_osr(grid.wkt)
        d = gdc.epsg_osr(grid.epsg)
        assert gdc.matching_spatref(a, b)
        assert gdc.matching_spatref(a, c)
        assert gdc.matching_spatref(a, d)


class TestExtent:
    """GDAL Common Extent class specific tests"""

    def test_extent_properties(self, grid):
        expected = grid.extent
        # Default rounding when building an extent is to 10 digits
        extent = gdc.Extent(grid.extent)
        assert extent.xmin == expected[0]
        assert extent.ymin == expected[1]
        assert extent.xmax == expected[2]
        assert extent.ymax == expected[3]

    def test_extent_rounding(self, grid, ndigits=3):
        """Test building an extent with different rounding"""
        expected = grid.extent
        # Default rounding when building an extent is to 10 digits
        extent = gdc.Extent(grid.extent, ndigits)
        assert extent.xmin == round(expected[0], ndigits)
        assert extent.ymin == round(expected[1], ndigits)
        assert extent.xmax == round(expected[2], ndigits)
        assert extent.ymax == round(expected[3], ndigits)

    def test_extent_str(self, grid):
        extent = grid.extent
        expected = ' '.join(['{}'.format(x) for x in extent])
        assert str(gdc.Extent(extent)) == expected

    def test_extent_list(self, grid):
        extent = grid.extent
        assert list(gdc.Extent(extent)) == extent

    def test_extent_adjust_to_snap_round(self, grid):
        extent_mod = grid.extent[:]
        # Adjust test extent out to the rounding limits
        extent_mod[0] = extent_mod[0] + 0.49 * grid.cellsize
        extent_mod[1] = extent_mod[1] - 0.49 * grid.cellsize
        extent_mod[2] = extent_mod[2] - 0.49 * grid.cellsize
        extent_mod[3] = extent_mod[3] + 0.49 * grid.cellsize
        extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
            'ROUND', grid.snap_x, grid.snap_y, grid.cellsize))
        assert extent_mod == pytest.approx(grid.extent, 0.00000001)

    def test_extent_adjust_to_snap_expand(self, grid):
        extent_mod = grid.extent[:]
        # Shrink the test extent in almost a full cellsize
        extent_mod[0] = extent_mod[0] + 0.99 * grid.cellsize
        extent_mod[1] = extent_mod[1] + 0.51 * grid.cellsize
        extent_mod[2] = extent_mod[2] - 0.51 * grid.cellsize
        extent_mod[3] = extent_mod[3] - 0.99 * grid.cellsize
        extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
            'EXPAND', grid.snap_x, grid.snap_y, grid.cellsize))
        assert extent_mod == pytest.approx(grid.extent, 0.00000001)

    def test_extent_adjust_to_snap_shrink(self, grid):
        extent_mod = grid.extent[:]
        # Expand the test extent out almost a full cellsize
        extent_mod[0] = extent_mod[0] - 0.99 * grid.cellsize
        extent_mod[1] = extent_mod[1] - 0.51 * grid.cellsize
        extent_mod[2] = extent_mod[2] + 0.51 * grid.cellsize
        extent_mod[3] = extent_mod[3] + 0.99 * grid.cellsize
        extent_mod = list(gdc.Extent(extent_mod).adjust_to_snap(
            'SHRINK', grid.snap_x, grid.snap_y, grid.cellsize))
        assert extent_mod == pytest.approx(grid.extent, 0.00000001)

    @pytest.mark.parametrize("distance", [10, -10])
    def test_extent_buffer(self, distance, grid):
        expected = grid.extent[:]
        expected[0] = expected[0] - distance
        expected[1] = expected[1] - distance
        expected[2] = expected[2] + distance
        expected[3] = expected[3] + distance
        extent_mod = list(gdc.Extent(grid.extent).buffer(distance))
        assert extent_mod == pytest.approx(expected, 0.00000001)

    # def test_extent_split(self, grid):
    #     """List of extent terms (xmin, ymin, xmax, ymax)"""
    #     assert gdc.Extent(grid.extent).split() == grid.extent

    def test_extent_copy(self, grid):
        """Return a copy of the extent"""
        orig_extent = gdc.Extent(grid.extent)
        copy_extent = orig_extent.copy()
        # Modify the original extent
        orig_extent = orig_extent.buffer(10)
        # Check that the copy hasn't changed
        assert list(copy_extent) == grid.extent

    def test_extent_corner_points(self, grid):
        """Corner points in clockwise order starting with upper-left point"""
        expected = [
            (grid.extent[0], grid.extent[3]),
            (grid.extent[2], grid.extent[3]),
            (grid.extent[2], grid.extent[1]),
            (grid.extent[0], grid.extent[1])]
        assert gdc.Extent(grid.extent).corner_points() == expected

    def test_extent_ul_lr_swap(self, grid):
        """Copy of extent object reordered as xmin, ymax, xmax, ymin

        Some gdal utilities want the extent described using upper-left and
        lower-right points.
            gdal_translate -projwin ulx uly lrx lry
            gdal_merge -ul_lr ulx uly lrx lry

        """
        expected = [
            grid.extent[0], grid.extent[3],
            grid.extent[2], grid.extent[1]]
        assert list(gdc.Extent(grid.extent).ul_lr_swap()) == expected

    def test_extent_ogrenv_swap(self, grid):
        """Copy of extent object reordered as xmin, xmax, ymin, ymax

        OGR feature (shapefile) extents are different than GDAL raster extents
        """
        expected = [
            grid.extent[0], grid.extent[2],
            grid.extent[1], grid.extent[3]]
        assert list(gdc.Extent(grid.extent).ogrenv_swap()) == expected

    def test_extent_origin(self, grid):
        """Origin (upper-left corner) of the extent"""
        assert gdc.Extent(grid.extent).origin() == grid.origin

    def test_extent_center(self, grid):
        """Centroid of the extent"""
        assert gdc.Extent(grid.extent).center() == grid.center

    def test_extent_shape(self, grid):
        """Return number of rows and columns of the extent

        Args:
            cs (int): cellsize
        Returns:
            tuple of raster rows and columns
        """
        extent = gdc.Extent(grid.extent)
        assert extent.shape(cs=grid.cellsize) == grid.shape

    def test_extent_geo(self, grid):
        """Geo-tranform of the extent"""
        extent = gdc.Extent(grid.extent)
        assert extent.geo(cs=grid.cellsize) == grid.geo

    def test_extent_geometry(self, grid):
        """Check GDAL geometry by checking if WKT matches"""
        extent_wkt = gdc.Extent(grid.extent).geometry().ExportToWkt()

        # This is needed to match the float formatting in ExportToWkt()
        # Using 10 to match default rounding in Extent.__init__()
        expected = [
            "{} {} 0".format(
                '{:.10f}'.format(x).rstrip('0').rstrip('.'),
                '{:.10f}'.format(y).rstrip('0').rstrip('.'))
            for x, y in gdc.Extent(grid.extent).corner_points()]

        # First point is repeated in geometry
        expected = "POLYGON (({}))".format(','.join(expected + [expected[0]]))
        assert extent_wkt == expected

    def test_extent_intersect_point(self, grid):
        """"Test if Point XY intersects the extent"""
        extent = gdc.Extent(grid.extent)
        origin = grid.origin
        cs = grid.cellsize
        assert not extent.intersect_point([origin[0] - cs, origin[1] + cs])
        assert extent.intersect_point([origin[0], origin[1]])
        assert extent.intersect_point([origin[0] + cs, origin[1] - cs])
        # assert extent.intersect_point(xy) == expected
        # assert extent.intersect_point(xy) == expected

    """Other extent related functions/tests"""
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            [[0, 0, 20, 20], [10, 10, 30, 30], True],
            [[0, 0, 20, 20], [30, 30, 50, 50], False]
        ]
    )
    def test_extents_overlap(self, a, b, expected):
        """Test if two extents overlap"""
        assert gdc.extents_overlap(gdc.Extent(a), gdc.Extent(b)) == expected

    @pytest.mark.parametrize(
        "extent_list,expected",
        [
            [[[0, 0, 20, 20], [10, 10, 30, 30]], [0, 0, 30, 30]],
            [[[0, 0, 20, 20], [10, 10, 30, 30], [20, 20, 40, 40]], [0, 0, 40, 40]]
        ]
    )
    def test_union_extents(self, extent_list, expected):
        """Return the union of all input extents"""
        extent_list = [gdc.Extent(extent) for extent in extent_list]
        assert list(gdc.union_extents(extent_list)) == expected

    @pytest.mark.parametrize(
        "extent_list,expected",
        [
            [[[0, 0, 20, 20], [10, 10, 30, 30]], [10, 10, 20, 20]],
            [[[0, 0, 20, 20], [10, 0, 30, 20], [0, 10, 20, 30]], [10, 10, 20, 20]]
        ]
    )
    def test_intersect_extents(self, extent_list, expected):
        """Return the intersection of all input extents"""
        extent_list = [gdc.Extent(extent) for extent in extent_list]
        assert list(gdc.intersect_extents(extent_list)) == expected

    # def test_project_extent(self, input_extent, input_osr, output_osr, cellsize):
    #     """Project extent to different spatial reference / coordinate system

    #     Args:
    #         input_extent (): the input gdal_common.extent to be reprojected
    #         input_osr (): OSR spatial reference of the input extent
    #         output_osr (): OSR spatial reference of the desired output
    #         cellsize (): the cellsize used to calculate the new extent.
    #             This cellsize is in the input spatial reference

    #     Returns:
    #         tuple: :class:`gdal_common.extent` in the desired projection
    #     """
    #     assert False


class TestGeo:
    """GeoTransform specific tests"""
    def test_geo_cellsize(self, grid):
        """Test getting cellsize tuple (x, y) from geo transform"""
        assert gdc.geo_cellsize(
            grid.geo, x_only=False) == (grid.cellsize, -grid.cellsize)

    def test_geo_cellsize_x_only(self, grid):
        """Test getting single cellsize value from geo transform"""
        assert gdc.geo_cellsize(grid.geo, x_only=True) == grid.cellsize

    def test_geo_origin(self, grid):
        """Test getting origin (upper left corner) from geo transform"""
        assert gdc.geo_origin(grid.geo) == grid.origin

    def test_geo_extent(self, grid):
        """Test getting extent from geo transform"""
        assert list(gdc.geo_extent(
            grid.geo, grid.rows, grid.cols)) == grid.extent

    def test_round_geo(self, grid, ndigits=3):
        """Round the values of a geotransform to n digits"""
        expected = tuple(round(x, ndigits) for x in grid.geo)
        assert gdc.round_geo(grid.geo, ndigits) == expected

    def test_geo_2_ee_transform(self, grid):
        """Test converting GDAL geo transform to EE crs transform"""
        assert gdc.geo_2_ee_transform(grid.geo) == grid.transform

    def test_ee_transform_2_geo(self, grid):
        """Test converting EE crs transform to GDAL geo transform"""
        assert gdc.ee_transform_2_geo(grid.transform) == grid.geo

    # def test_array_offset_geo(self, grid):
    #     """Return sub_geo that is offset from full_geo"""
    #     assert gdc.array_offset_geo(test_geo, x_offset, y_offset) == expected

    # def test_array_geo_offsets(self, grid):
    #     """Return x/y offset of a gdal.geotransform based on another gdal.geotransform"""
    #     assert gdc.array_geo_offsets(full_geo, sub_geo, grid.cellsize) == expected


class TestGeoJson:
    @pytest.fixture(scope='class')
    def points(self, grid):
        """Convert grid corner points to GeoJson coordinates"""
        return list(map(list, gdc.Extent(grid.extent).corner_points()))

    def test_json_reverse_polygon(self, points):
        """Reverse the point order from counter-clockwise to clockwise"""
        json_geom = {'type': 'Polygon', 'coordinates': [points]}
        expected = {'type': 'Polygon', 'coordinates': [points[::-1]]}
        assert gdc.json_reverse_func(json_geom) == expected

    def test_json_reverse_multipolygon(self, points):
        """Reverse the point order from counter-clockwise to clockwise"""
        json_geom = {'type': 'MultiPolygon', 'coordinates': [[points]]}
        expected = {'type': 'MultiPolygon', 'coordinates': [[points[::-1]]]}
        assert gdc.json_reverse_func(json_geom) == expected

    def test_json_strip_z_polygon(self, points):
        """Strip Z value from coordinates"""
        json_geom = {
            'type': 'Polygon', 'coordinates': [[p + [1.0] for p in points]]}
        expected = {'type': 'Polygon', 'coordinates': [points]}
        assert gdc.json_strip_z_func(json_geom) == expected

    def test_json_strip_z_multipolygon(self, points):
        """Strip Z value from coordinates"""
        json_geom = {
            'type': 'MultiPolygon',
            'coordinates': [[[p + [1.0] for p in points]]]}
        expected = {'type': 'MultiPolygon', 'coordinates': [[points]]}
        assert gdc.json_strip_z_func(json_geom) == expected


class TestFeature:
    """Feature specific tests"""
    @pytest.fixture(scope='class')
    def feature(self, grid):
        """Convert grid corner points to GeoJson coordinates"""
        return Feature(grid)

    def test_feature_lyr_osr(self, feature):
        assert gdc.matching_spatref(
            gdc.feature_lyr_osr(feature.lyr), feature.osr)

    def test_feature_ds_osr(self, feature):
        assert gdc.matching_spatref(
            gdc.feature_ds_osr(feature.ds), feature.osr)

    # def test_feature_path_osr(self, feature):
    #     assert gdc.matching_spatref(
    #         gdc.feature_path_osr(feature.path), feature.osr)

    def test_feature_lyr_extent(self, feature):
        assert list(gdc.feature_lyr_extent(feature.lyr)) == feature.extent

    def test_feature_lyr_fields(self, feature):
        assert gdc.feature_lyr_fields(feature.lyr) == ['Name', 'Lat', 'Lon']

    def test_feature_ds_fields(self, feature):
        assert gdc.feature_ds_fields(feature.ds) == ['Name', 'Lat', 'Lon']

    # def test_feature_path_fields(self, feature):
    #     assert gdc.feature_path_fields(feature.path) == []

    # def test_shapefile_2_geom_list_func(input_path, zone_field=None,
    #                                     reverse_flag=False, simplify_flag=False):
    #     """Return a list of feature geometries in the shapefile

    #     Also return the FID and value in zone_field
    #     FID value will be returned if zone_field is not set or does not exist
    #     """
    #     assert False



class TestRaster:
    """Raster specific tests"""

    @pytest.fixture(scope='class')
    def memory_raster(request, grid):
        """Generic in-memory raster for testing properties"""
        return Raster(grid)

    @pytest.fixture(scope='class')
    def file_raster(request, grid, tmpdir_factory):
        """Generic file raster for testing raster_path* functions"""
        filename = str(tmpdir_factory.getbasetemp().join('test.img'))
        params = {'filename': filename}
        return Raster(grid, **params)

    def test_raster_ds_geo(self, memory_raster):
        assert gdc.raster_ds_geo(memory_raster.ds) == memory_raster.geo

    def test_raster_ds_extent(self, memory_raster):
        assert list(gdc.raster_ds_extent(
            memory_raster.ds)) == memory_raster.extent

    def test_raster_ds_shape(self, memory_raster):
        assert gdc.raster_ds_shape(memory_raster.ds) == memory_raster.shape

    def test_raster_path_shape(self, file_raster):
        """Test wrapper to raster_ds_shape"""
        assert gdc.raster_path_shape(file_raster.path) == file_raster.shape

    # Default raster is: {
    #     'filename': '', 'dtype': np.float32, 'r_nodata': 'DEFAULT'
    #     'v_min': 0, 'v_max': 10, 'v_nodata': 10}
    # Modify default to try out different dtypes, nodata, and raster types
    @pytest.mark.parametrize(
        "params",
        [
            {'dtype': np.uint8},
            {'dtype': np.float32, 'r_nodata': 'DEFAULT'},
            {'dtype': np.float32, 'r_nodata': 10},
            # {'dtype': np.float32, 'filename': 'test.img'},
            # {'dtype': np.float32, 'filename': 'test.tif'},
        ])
    def test_raster_ds_to_array(self, params, grid):
        """Test reading NumPy array from raster

        Output array size will match the mask_extent if mask_extent is set

        Args:
            input_raster_ds (): opened raster dataset as gdal raster
            mask_extent (): subset extent of the raster if desired
            fill_value (float): Value to Initialize empty array with
        """
        raster = Raster(grid, **params)
        raster_array = gdc.raster_ds_to_array(
            raster.ds, return_nodata=False)
        assert np.array_equal(
            raster_array[np.isfinite(raster_array)],
            raster.array[np.isfinite(raster.array)])

    @pytest.mark.parametrize(
        "params",
        [
            {'dtype': np.uint8},
            # {'dtype': np.uint8, 'r_nodata': 10},
            # {'dtype': np.uint8, 'r_nodata': None},
            {'dtype': np.float32},
            # {'dtype': np.float32, 'v_nodata': 10},
            # {'dtype': np.float32, 'r_nodata': 10},
            # {'dtype': np.float32, 'r_nodata': None},
            # {'dtype': np.float32, 'r_nodata': 'DEFAULT'},
            # {'dtype': np.float32, 'r_nodata': 10},
            # {'dtype': np.float32, 'filename': 'test.img'},
            # {'dtype': np.float32, 'filename': 'test.tif'},
        ])
    def test_raster_ds_to_array_return_nodata(self, params, grid):
        """Test reading raster array and nodata value"""
        raster = Raster(grid, **params)
        raster_array, raster_nodata = gdc.raster_ds_to_array(
            raster.ds, return_nodata=True)
        # print('')
        # print(raster_array)
        # print(raster_nodata)
        # print(raster.array)
        # print(raster.nodata)
        assert np.array_equal(
            raster_array[np.isfinite(raster_array)],
            raster.array[np.isfinite(raster.array)])

        # Nodata value is always "nan" for float types
        if raster.dtype in [np.float32, np.float64]:
            assert np.isnan(raster_nodata)
        else:
            assert raster_nodata == raster.nodata
        # assert False

    @pytest.mark.parametrize(
        "params",
        [
            {'filename': ''},
            {'filename': '', 'v_nodata': None, 'r_nodata': None},
            {'filename': 'test.img', 'v_nodata': None, 'r_nodata': None},
            {'filename': 'test.tif', 'v_nodata': None, 'r_nodata': None},
        ])
    def test_raster_ds_to_array_default_nodata(self, params, grid, tmpdir):
        """Test reading raster with default_nodata_value set

        If the raster does not have a nodata value,
            use fill_value as the nodata value
        """
        if 'nodata' in params.keys() and params['nodata'] is not None:
            pytest.skip('Only test if raster does not have a nodata value set')
        else:
            # Add temporary folder to raster file name
            # If I don't make a copy, then params ends up in a weird state
            params_copy = params.copy()
            if 'filename' in params.keys() and params['filename']:
                params_copy['filename'] = str(tmpdir.join(params['filename']))

            raster = Raster(grid, **params_copy)
            nodata = 10
            raster_array = gdc.raster_ds_to_array(
                raster.ds, default_nodata_value=nodata,
                return_nodata=False)
            expected = raster.array[:]
            expected[expected == nodata] = np.nan
            assert np.array_equal(
                raster_array[np.isfinite(raster_array)],
                expected[np.isfinite(expected)])

    # def test_raster_ds_to_array_mask_extent(self, params, grid):
    #     """Test reading raster with mask_extent"""
    #     raster = Raster(grid, **params)
    #     # raster_array, raster_nodata = gdc.raster_ds_to_array(
    #     #     raster.ds, mask_extent=[], return_nodata=False)
    #     assert np.array_equal(
    #         raster_array[np.isfinite(raster_array)],
    #         raster.array[np.isfinite(raster.array)])

    def test_raster_to_array(self, file_raster):
        """Test wrapper to raster_ds_to_array"""
        raster_array = gdc.raster_ds_to_array(
            file_raster.ds, return_nodata=False)
        assert np.array_equal(
            raster_array[np.isfinite(raster_array)],
            file_raster.array[np.isfinite(file_raster.array)])

    # def test_raster_ds_set_nodata(self, raster, input_nodata):
    #     """Set raster dataset nodata value for all bands"""
    #     assert gdc.raster_ds_set_nodata(raster.ds, input_nodata)

    # def test_raster_path_set_nodata(self, raster, input_nodata):
    #     """Set raster nodata value for all bands"""
    #     assert gdc.raster_path_set_nodata(raster.path, input_nodata)


class TestArray:
    """Array specific tests"""
    pass

    # def test_project_array(self, input_array, resampling_type,
    #                        input_osr, input_cs, input_extent,
    #                        output_osr, output_cs, output_extent,
    #                        output_nodata=None):
    #     """Project a NumPy array to a new spatial reference

    #     This function doesn't correctly handle masked arrays
    #     Must pass output_extent & output_cs to get output raster shape
    #     There is not enough information with just output_geo and output_cs

    #     Args:
    #         input_array (array: :class:`numpy.array`):
    #         resampling_type ():
    #         input_osr (:class:`osr.SpatialReference):
    #         input_cs (int):
    #         input_extent ():
    #         output_osr (:class:`osr.SpatialReference):
    #         output_cs (int):
    #         output_extent ():
    #         output_nodata (float):

    #     Returns:
    #         array: :class:`numpy.array`
    #     """

    #     assert False
