#--------------------------------
# Name:         ee_common.py
# Purpose:      Common EarthEngine support functions
# Python:       3.6
#--------------------------------

from builtins import input
import datetime
import logging
import math
import pprint
import sys

import ee


ee.Initialize()

system_properties = ['system:index', 'system:time_start']

refl_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
refl_toa_bands = [b + '_toa' for b in refl_bands]
refl_sur_bands = [b + '_sur' for b in refl_bands]


def show_thumbnail(ee_image):
    """Show the EarthEngine image thumbnail in a window"""
    output_url = ee_image.getThumbUrl({'format': 'jpg', 'size': '600'})
    logging.debug(output_url)

    # import webbrowser
    # webbrowser.open(output_url)

    import io
    # import Image, ImageTk
    from PIL import Image, ImageTk
    import Tkinter as tk
    import urllib2
    window = tk.Tk()
    output_file = Image.open(io.BytesIO(urllib2.urlopen(output_url).read()))
    output_photo = ImageTk.PhotoImage(output_file)
    label = tk.Label(window, image=output_photo)
    label.pack()
    window.mainloop()


class Landsat():
    mosaic_options = ['min', 'max', 'median', 'mean', 'mosaic']
    cellsize = 30

    def __init__(self, args):
        """Initialize the class with the user specified arguments

        All argument strings should be lower case

        Args: dictionary with the following key/values
            refl_source (str): At-surface reflectance source
                Choices: 'tasumi', 'ledaps', 'lasrc' (same as ledaps)
            fmask_flag (bool): if True, mask Fmask cloud, shadow, and snow pixels
            acca_flag (bool): if True, mask pixels with clouds scores > 50
            start_date (str): ISO format start date (YYYY-MM-DD)
            end_date (str): ISO format end date (YYYY-MM-DD) (inclusive)
            start_year (int): start year
            end_year (int): end year
            start_month (int): start month
            end_month (int): end month
            start_doy (int): start day of year
            end_doy (int): end day of year
            zone_geom (ee.Geometry): apply filterBounds using this geometry
            scene_id_keep_list (list): SCENE_IDs to explicitly include
                SCENE_IDs do not include version or downlink station
                Example: "LT05_041032_1984214"
            scene_id_skip_list (list): SCENE_IDs to explicitly skip/exclude
                SCENE_IDs do not include version or downlink station
                Example: "LT05_041032_1984214"
            path_keep_list (list): Landsat path numbers (as int)
            row_keep_list (list): Landsat row numbers (as int)
            tile_keep_list (list): Landsat WRS2 tile strings ('p042r33')
            tile_geom (ee.Geometry):
            adjust_method (str): Adjust Landsat red and NIR bands.
                Choices: 'etm_2_oli' or 'oli_2_etm'.
                This could probably be simplifed to a flag.
                This flag is passed through and not used in this function
            mosaic_method (str): Method for mosaicing overlapping images
                Choices: 'mean', 'median', 'mosaic', 'min', 'max'
            refl_sur_method (str): at-surface reflectance method
                Choices: 'tasumi' or 'usgs_sr'
            products (list): Landsat bands to compute/return
            landsat4_flag (bool): if True, include Landsat 4 images
            landsat5_flag (bool): if True, include Landsat 5 images
            landsat7_flag (bool): if True, include Landsat 7 images
            landsat8_flag (bool): if True, include Landsat 8 images
            landsat9_flag (bool): if True, include Landsat 9 images

        """
        arg_list = [
            'fmask_flag', 'acca_flag',
            'start_date', 'end_date', 'start_year', 'end_year',
            'start_month', 'end_month', 'start_doy', 'end_doy',
            'zone_geom', 'scene_id_keep_list', 'scene_id_skip_list',
            'path_keep_list', 'row_keep_list', 'tile_keep_list', 'tile_geom',
            'refl_sur_method', 'adjust_method', 'mosaic_method', 'products',
            'landsat4_flag', 'landsat5_flag', 'landsat7_flag', 'landsat8_flag',
            'landsat9_flag'
        ]
        int_args = [
            'start_year', 'end_year', 'start_month', 'end_month',
            'start_doy', 'end_doy'
        ]
        # list_args = [
        #     'products', 'scene_id_keep_list', 'scene_id_skip_list',
        #     'path_keep_list', 'row_keep_list']

        # Set default products list if it was not set
        if 'products' not in args:
            args['products'] = []

        # # Set start and end date if they are not set
        # # This is needed for selecting Landsat collections below
        # if not args['start_date'] and args['start_year']:
        #     args['start_date'] = '{}-01-01'.format(args['start_year'])
        # elif not args['start_date'] and args['start_year']:
        #     args['start_date'] = '1982-01-01'
        # if not args['end_date'] and args['end_year']:
        #     args['end_date'] = '{}-12-31'.format(args['end_year'])
        # elif not args['end_date'] and args['end_date']:
        #     args['end_date'] = datetime.datetime.now().strftime('%Y-%m-%d')

        # logging.debug('  Init Args')
        for key in arg_list:
            try:
                if str(key) in int_args:
                    value = int(args[key])
                else:
                    value = args[key]
            except KeyError:
                # Argument was not passed in or set
                value = None
            except TypeError:
                # Argument is not integer type
                value = None
            setattr(self, str(key), value)
            # if key not in ['zone_geom']:
            #     logging.debug('  {}: {}'.format(key, value))

        # Landsat list should not be directly set by the user
        # It will be computed from the flags
        self.set_landsat_from_flags()

        today = datetime.date.today().isoformat()
        self.dates = {
            'LT04': {'start': '1982-01-01', 'end': '1993-12-31'},
            'LT05': {'start': '1984-01-01', 'end': '2011-12-31'},
            'LE07': {'start': '1999-01-01', 'end': '2021-09-27'}, #added end for LS7 
            'LC08': {'start': '2013-03-24', 'end': today},
            'LC09': {'start': '2021-10-31', 'end': today}
        }

    def set_landsat_from_flags(self):
        """Set Landsat type list based on INI flags"""
        landsat_list = []
        if self.landsat4_flag:
            landsat_list.append('LT04')
        if self.landsat5_flag:
            landsat_list.append('LT05')
        if self.landsat7_flag:
            landsat_list.append('LE07')
        if self.landsat8_flag:
            landsat_list.append('LC08')
        if self.landsat9_flag:
            landsat_list.append('LC09')
        self._landsat_list = sorted(landsat_list)

    def set_landsat_from_scene_id(self):
        """Set Landsat type list based on SCENE_ID keep list"""
        if self.scene_id_keep_list:
            self._landsat_list = sorted(list(set(
                [str(x[:4]) for x in self.scene_id_keep_list])))

    def set_tiles_from_scene_id(self):
        # Remove path/rows that aren't needed
        if self.scene_id_keep_list:
            self.path_keep_list = sorted(list(set([
                int(x[5:8]) for x in self.scene_id_keep_list])))
            self.row_keep_list = sorted(list(set([
                int(x[8:11]) for x in self.scene_id_keep_list])))
            # Tile keep list is not supported yet in get_collection()
            # self.tile_keep_list = sorted(list(set([
            #     'p{:03d}r{:03d}'.format(x[5:8], x[8:11])
            #     for x in self.scene_id_keep_list])))

    def get_image(self, landsat, year, doy, path=None, row=None):
        """Return a single Landsat image

        Mosaic images from different rows from the same date (same path)

        Parameters
        ----------
        landsat : str
        year : int
        doy : int
            Day of year.
        path : int
            Landsat path number.
        row : int
            Landsat row number.

        Returns
        -------
        ee.Image

        """
        image_start_dt = datetime.datetime.strptime(
            '{:04d}_{:03d}'.format(int(year), int(doy)), '%Y_%j')
        image_end_dt = image_start_dt + datetime.timedelta(days=1)

        # Adjust the default keyword arguments for a single image date
        self.start_date = image_start_dt.date().isoformat()
        self.end_date = image_end_dt.date().isoformat()
        # self.start_year = year
        # self.end_dear = year
        # self.start_doy = doy
        # self.end_doy = doy
        if path:
            self.path_keep_list = [int(path)]
        if row:
            self.row_keep_list = [int(row)]
        # if path and row:
        #     self.pathrow_keep_list = [
        #         'p{:03d}r{:03d}'.format(int(path), int(row))]

        # Landsat collection for a single date
        landsat_coll = self.get_collection()
        return ee.Image(landsat_coll.first())

    def get_collection(self):
        """Build and filter a full Landsat collection

        Parameters
        ----------
        args : dict
            Keyword arguments for get_landsat_collection.

        Returns
        -------
        ee.ImageCollection

        """

        # Process each Landsat type and append to the output collection
        output_coll = ee.ImageCollection([])
        for landsat in self._landsat_list:
            # Assume ieration will be controlled by changing start_date and end_date
            # Skip Landsat collections that are outside these date ranges
            if self.end_date and self.end_date < self.dates[landsat]['start']:
                continue
            elif self.start_date and self.start_date > self.dates[landsat]['end']:
                continue
            # Is it necessary or helpful to check year also?
            elif (self.end_year and
                    self.end_year < int(self.dates[landsat]['start'][:4])):
                continue
            elif (self.start_year and
                    self.start_year > int(self.dates[landsat]['end'][:4])):
                continue
            # logging.debug('  Landsat: {}'.format(landsat))

            # TOA reflectance collection
            # if landsat in ['LE07', 'LC08']:
            #     toa_name = 'LANDSAT/{}/C02/T1_RT_TOA'.format(landsat)
            if landsat in ['LT05', 'LE07', 'LC08', 'LC09']:
                # Per convo w/ charles, no need for RT data, collecting all TOA data from T1_TOA
                toa_name = 'LANDSAT/{}/C02/T1_TOA'.format(landsat)
            elif landsat in ['LT04']:
                # DEADBEEF - Landsat 4 Collection 1 TOA is not available yet
                # Mar 2022, collection 2 TOA not available in GEE (GS)
                continue
            toa_coll = ee.ImageCollection(toa_name)

            # At-surface reflectance collection
            sr_name = 'LANDSAT/{}/C02/T1_L2'.format(landsat)
            if self.refl_sur_method == 'tasumi':
                # Tasumi SR will be computed from TOA after filtering
                sur_coll = ee.ImageCollection(toa_name)
            elif self.refl_sur_method == 'usgs_sr':
                sur_coll = ee.ImageCollection(sr_name)
            else:
                logging.error(
                    '\nERROR: Unsupported Landsat/reflectance combination, exiting\n'
                    '  Landsat: {}  Reflectance: {}'.format(
                        landsat, self.refl_sur_method))
                sys.exit()

            # Filter non-L1T/L1TP images
            # There are a couple of non-L1TP images in LE07 collection 1
            # if landsat in ['LE07']:
            #     toa_coll = toa_coll.filterMetadata(
            #         'DATA_TYPE', 'equals', 'L1TP')
                # sur_coll = sur_coll.filterMetadata(
                #     'DATA_TYPE', 'equals', 'L1TP')
            # Exclude 2012 Landsat 5 images
            if landsat in ['LT05']:
                toa_coll = toa_coll.filter(
                    ee.Filter.calendarRange(1984, 2011, 'year'))
                sur_coll = sur_coll.filter(
                    ee.Filter.calendarRange(1984, 2011, 'year'))
            # Exclude early Landsat 8 images (operational on April 11th, 2013)
            elif landsat in ['LC08']:
                toa_coll = toa_coll.filter(ee.Filter.gt(
                    'system:time_start', ee.Date('2013-03-24').millis()))
                sur_coll = sur_coll.filter(ee.Filter.gt(
                    'system:time_start', ee.Date('2013-03-24').millis()))

            # Filter by date
            if self.start_date and self.end_date:
                # End date is inclusive but filterDate is exclusive
                end_date = (
                    datetime.datetime.strptime(self.end_date, '%Y-%m-%d') +
                    datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                toa_coll = toa_coll.filterDate(self.start_date, end_date)
                sur_coll = sur_coll.filterDate(self.start_date, end_date)
            # Filter by year
            if self.start_year and self.end_year:
                year_filter = ee.Filter.calendarRange(
                    self.start_year, self.end_year, 'year')
                toa_coll = toa_coll.filter(year_filter)
                sur_coll = sur_coll.filter(year_filter)
            # Filter by month
            if ((self.start_month and self.start_month != 1) or
                    (self.end_month and self.end_month != 12)):
                month_filter = ee.Filter.calendarRange(
                    self.start_month, self.end_month, 'month')
                toa_coll = toa_coll.filter(month_filter)
                sur_coll = sur_coll.filter(month_filter)
            # Filter by day of year
            if ((self.start_doy and self.start_doy != 1) or
                    (self.end_doy and self.end_doy != 365)):
                doy_filter = ee.Filter.calendarRange(
                    self.start_doy, self.end_doy, 'day_of_year')
                toa_coll = toa_coll.filter(doy_filter)
                sur_coll = sur_coll.filter(doy_filter)

            if self.path_keep_list:
                toa_coll = toa_coll.filter(
                    ee.Filter.inList('WRS_PATH', self.path_keep_list))
                sur_coll = sur_coll.filter(
                    ee.Filter.inList('WRS_PATH', self.path_keep_list))
            if self.row_keep_list:
                toa_coll = toa_coll.filter(
                    ee.Filter.inList('WRS_ROW', self.row_keep_list))
                sur_coll = sur_coll.filter(
                    ee.Filter.inList('WRS_ROW', self.row_keep_list))
            if self.tile_geom:
                toa_coll = toa_coll.filterBounds(self.tile_geom)
                sur_coll = sur_coll.filterBounds(self.tile_geom)

            # Filter by geometry
            if self.zone_geom:
                toa_coll = toa_coll.filterBounds(self.zone_geom)
                sur_coll = sur_coll.filterBounds(self.zone_geom)

            # Filter by SCENE_ID
            if self.scene_id_keep_list:
                scene_id_keep_filter = ee.Filter.inList(
                    'system:index', self.scene_id_keep_list)
                toa_coll = toa_coll.filter(scene_id_keep_filter)
                sur_coll = sur_coll.filter(scene_id_keep_filter)
            if self.scene_id_skip_list:
                scene_id_skip_filter = ee.Filter.inList(
                    'system:index', self.scene_id_skip_list).Not()
                toa_coll = toa_coll.filter(scene_id_skip_filter)
                sur_coll = sur_coll.filter(scene_id_skip_filter)

            # Add ACCA band (must be done before bands are renamed below)
            toa_coll = toa_coll.map(landsat_acca_band_func)

            # Extract Fmask band from QA band
            # toa_coll = toa_coll.map(landsat_bqa_fmask_func)
            # updated 'fmask' for collection 2 from QA_PIXEL
            toa_coll = toa_coll.map(landsat_col2_cloudmask_func)

            # Modify landsat collections to have same band names
            if landsat in ['LT04', 'LT05']:
                toa_coll = toa_coll.map(landsat5_toa_band_func)
                sur_coll = sur_coll.map(landsat5_sur_band_func)
            elif landsat in ['LE07']:
                toa_coll = toa_coll.map(landsat7_toa_band_func)
                sur_coll = sur_coll.map(landsat7_sur_band_func)
            elif landsat in ['LC08']:
                toa_coll = toa_coll.map(landsat8_toa_band_func)
                sur_coll = sur_coll.map(landsat8_sur_band_func)
            elif landsat in ['LC09']: 
                toa_coll = toa_coll.map(landsat9_toa_band_func)
                sur_coll = sur_coll.map(landsat9_sur_band_func)

            # Compute Tasumi SR (if necessary) after filtering collections
            # DEADBEEF - This may need to be moved before band functions
            if self.refl_sur_method == 'tasumi':
                sur_coll = sur_coll.map(self.tasumi_sr_func)

            # Apply OLI 2 ETM or ETM 2 OLI adjustments
            # Removed 7 Mar 2022; collection 2 is already cross-sensor calibrated
            # if self.refl_sur_method == 'tasumi':
            #     sur_coll = sur_coll.map(self.tasumi_sr_adjust_func)
            # elif self.refl_sur_method == 'usgs_sr':
                # sur_coll = sur_coll.map(self.usgs_sr_adjust_func)

            # Join the TOA and SR collections
            scene_id_filter = ee.Filter.equals(
                leftField='system:index', rightField='system:index')
            landsat_coll = ee.ImageCollection(
                ee.Join.saveFirst('sur').apply(
                    toa_coll, sur_coll, scene_id_filter))

            # Add SR bands from joined property
            landsat_coll = ee.ImageCollection(
                landsat_coll.map(landsat_sur_band_func))

            # Apply cloud masks
            if self.fmask_flag:
                landsat_coll = landsat_coll.map(landsat_fmask_cloud_mask_func)
            if self.acca_flag:
                landsat_coll = landsat_coll.map(landsat_acca_cloud_mask_func)

            # # Get the output image URL
            # output_url = ee.Image(landsat_coll.first()) \
            #     .select(['red_sur', 'green_sur', 'blue_sur']) \
            #     .visualize(min=[0, 0, 0], max=[0.4, 0.4, 0.4]) \
            #     .getThumbUrl({'format': 'png', 'size': '600'})
            # # This would load the image in your browser
            # import webbrowser
            # webbrowser.open(output_url)
            # # webbrowser.read(output_url)

            # Compute derived images
            landsat_coll = landsat_coll.map(self.landsat_images_func)
            # pprint.pprint(ee.Image(landsat_coll.first()).getInfo())
            # logging.info('{} {}'.format(landsat, landsat_coll.size().getInfo()))
            # logging.info('{}'.format(', '.join(sorted(
            #     landsat_coll.aggregate_histogram('SCENE_ID').getInfo().keys()))))

            # Mosaic overlapping images
            if (self.mosaic_method and
                    self.mosaic_method in self.mosaic_options):
                landsat_coll = mosaic_landsat_images(
                    landsat_coll, self.mosaic_method)
            # logging.info('{} {}'.format(landsat, landsat_coll.size().getInfo()))
            # logging.info('{}'.format(', '.join(sorted(
            #     landsat_coll.aggregate_histogram('SCENE_ID').getInfo().keys()))))
            # input('ENTER')

            # Merge Landsat specific collection into output collection
            output_coll = ee.ImageCollection(
                output_coll.merge(landsat_coll))

        # logging.info('{}'.format(', '.join(sorted(
        #     output_coll.aggregate_histogram('SCENE_ID').getInfo().keys()))))
        return output_coll

    def landsat_images_func(self, input_image):
        """Calculate Landsat products

        Send Landsat ROW number back as an image for determining "dominant"
            row in zones that overlap multiple images.

        Parameters
        ----------
        input_image : ee.Image
            Landsat merged TOA/SR image.

        Returns
        -------
        ee.Image

        """

        # Eventually use Fmask band to set common area instead
        # refl_toa = refl_toa_orig.updateMask(
        #     refl_toa_orig.select(['fmask']).gte(0))

        # Brightness temperature must be > 250 K
        # refl_toa = refl_toa.updateMask(refl_toa.select(['lst']).gt(250))

        # Output individual TOA and SR reflectance bands before renaming
        output_images = []
        for band in refl_toa_bands:
            if band in self.products:
                output_images.append(input_image.select([band]))
        for band in refl_sur_bands:
            if band in self.products:
                output_images.append(input_image.select([band]))

        # Separate TOA and SR reflectance composites
        # Rename bands back to standard reflectance band names (without "_toa")
        # This is needed so that index functions will work for either composite
        refl_toa = input_image.select(refl_toa_bands, refl_bands)
        refl_sur = input_image.select(refl_sur_bands, refl_bands)

        # At-surface albedo
        if 'albedo_sur' in self.products:
            albedo_sur = ee.Image(landsat_albedo_func(refl_sur)) \
                .rename(['albedo_sur'])
            output_images.append(albedo_sur)

        # NDVI (used to compute LAI)
        if ('ndvi_toa' in self.products or 'lai_toa' in self.products or
                'ts' in self.products):
            ndvi_toa = refl_toa.normalizedDifference(['nir', 'red']) \
                .rename(['ndvi_toa'])
            output_images.append(ndvi_toa)
        if 'ndvi_sur' in self.products or 'lai_sur' in self.products:
            ndvi_sur = refl_sur.normalizedDifference(['nir', 'red']) \
                .rename(['ndvi_sur'])
            output_images.append(ndvi_sur)

        # EVI (used to compute ET*)
        if ('evi_sur' in self.products or
                any([True for p in self.products if 'etstar_' in p]) or
                any([True for p in self.products if 'etg_' in p])):
            evi_sur = ee.Image(landsat_evi_func(refl_sur)) \
                .rename(['evi_sur'])
            output_images.append(evi_sur)

        # LAI (for computing Ts) (Empirical function from Allen et al 2007)
        if 'lai_toa' in self.products or 'ts' in self.products:
            lai_toa = ee.Image(ndvi_lai_func(ndvi_toa)) \
                .rename(['lai_toa'])
            output_images.append(lai_toa)
        if 'lai_sur' in self.products:
            lai_sur = ee.Image(ndvi_lai_func(ndvi_sur)) \
                .rename(['lai_sur'])
            output_images.append(lai_sur)

        # MSAVI
        if 'msavi_toa' in self.products:
            msavi_toa = ee.Image(landsat_msavi_func(refl_toa)) \
                .rename(['msavi_toa'])
            output_images.append(msavi_toa)
        if 'msavi_sur' in self.products:
            msavi_sur = ee.Image(landsat_msavi_func(refl_sur)) \
                .rename(['msavi_sur'])
            output_images.append(msavi_sur)

        # NDWI - McFeeters 1996
        if 'ndwi_green_nir_toa' in self.products:
            ndwi_green_nir_toa = refl_toa \
                .normalizedDifference(['green', 'nir']) \
                .rename(['ndwi_green_nir_toa'])
            output_images.append(ndwi_green_nir_toa)
        if 'ndwi_green_nir_sur' in self.products:
            ndwi_green_nir_sur = refl_sur \
                .normalizedDifference(['green', 'nir']) \
                .rename(['ndwi_green_nir_sur'])
            output_images.append(ndwi_green_nir_sur)

        # NDWI - Xu 2006 (MNDWI) doi: 10.1080/01431160600589179
        # Equivalent to NDSI Hall et al 1995 and 1998
        # http://modis-snow-ice.gsfc.nasa.gov/uploads/pap_dev95.pdf
        # http://modis-snow-ice.gsfc.nasa.gov/uploads/pap_assmnt98.pdf
        if 'ndwi_green_swir1_toa' in self.products:
            ndwi_green_swir1_toa = refl_toa \
                .normalizedDifference(['green', 'swir1']) \
                .rename(['ndwi_green_swir1_toa'])
            output_images.append(ndwi_green_swir1_toa)
        if 'ndwi_green_swir1_sur' in self.products:
            ndwi_green_swir1_sur = refl_sur \
                .normalizedDifference(['green', 'swir1']) \
                .rename(['ndwi_green_swir1_sur'])
            output_images.append(ndwi_green_swir1_sur)

        # NDWI - Gao 1996 doi: 10.1016/S0034-4257(96)00067-3
        # Inverse of NDSI (Soil) in Rogers & Keraney 2004
        if 'ndwi_nir_swir1_toa' in self.products:
            ndwi_nir_swir1_toa = refl_toa \
                .normalizedDifference(['nir', 'swir1']) \
                .rename(['ndwi_nir_swir1_toa'])
            output_images.append(ndwi_nir_swir1_toa)
        if 'ndwi_nir_swir1_sur' in self.products:
            ndwi_nir_swir1_sur = refl_sur \
                .normalizedDifference(['nir', 'swir1']) \
                .rename(['ndwi_nir_swir1_sur'])
            output_images.append(ndwi_nir_swir1_sur)

        # NDWI - Allen 2007
        # Return this NDWI as the default ndwi_sur and ndwi_toa below
        if 'ndwi_swir1_green_toa' in self.products:
            ndwi_swir1_green_toa = refl_toa \
                .normalizedDifference(['swir1', 'green']) \
                .rename(['ndwi_swir1_green_toa'])
            output_images.append(ndwi_swir1_green_toa)
        if 'ndwi_swir1_green_sur' in self.products:
            ndwi_swir1_green_sur = refl_sur \
                .normalizedDifference(['swir1', 'green']) \
                .rename(['ndwi_swir1_green_sur'])
            output_images.append(ndwi_swir1_green_sur)

        if 'ndwi_toa' in self.products:
            ndwi_toa = refl_toa \
                .normalizedDifference(['swir1', 'green']) \
                .rename(['ndwi_toa'])
            output_images.append(ndwi_toa)
        if 'ndwi_sur' in self.products:
            ndwi_sur = refl_sur \
                .normalizedDifference(['swir1', 'green']) \
                .rename(['ndwi_sur'])
            output_images.append(ndwi_sur)

        # SAVI
        if 'savi_toa' in self.products:
            savi_toa = ee.Image(landsat_savi_func(refl_toa)) \
                .rename(['savi_toa'])
            output_images.append(savi_toa)
        if 'savi_sur' in self.products:
            savi_sur = ee.Image(landsat_savi_func(refl_sur)) \
                .rename(['savi_sur'])
            output_images.append(savi_sur)

        # Surface temperature
        if 'ts' in self.products:
            ts = ee.Image(ts_func(
                ts_brightness=input_image.select('lst'),
                em_nb=em_nb_func(ndvi_toa, lai_toa),
                k1=ee.Number(refl_toa.get('k1_constant')),
                k2=ee.Number(refl_toa.get('k2_constant'))))
            output_images.append(ts)

        # Tasseled cap
        if 'tc_bright' in self.products:
            tc_bright = ee.Image(tc_bright_func(refl_toa))
            output_images.append(tc_bright)
        if 'tc_green' in self.products:
            tc_green = ee.Image(tc_green_func(refl_toa))
            output_images.append(tc_green)
        if 'tc_wet' in self.products:
            tc_wet = ee.Image(tc_wet_func(refl_toa))
            output_images.append(tc_wet)

        # Beamer ET* and ETg
        if 'etstar_mean' in self.products or 'etg_mean' in self.products:
            etstar_mean = ee.Image(etstar_func(evi_sur, etstar_type='mean')) \
                .rename(['etstar_mean'])
        if 'etstar_lpi' in self.products or 'etg_lpi' in self.products:
            etstar_lpi = ee.Image(etstar_func(evi_sur, etstar_type='lpi')) \
                .rename(['etstar_lpi'])
        if 'etstar_upi' in self.products or 'etg_upi' in self.products:
            etstar_upi = ee.Image(etstar_func(evi_sur, etstar_type='upi')) \
                .rename(['etstar_upi'])
        if 'etstar_lci' in self.products or 'etg_lci' in self.products:
            etstar_lci = ee.Image(etstar_func(evi_sur, etstar_type='lci')) \
                .rename(['etstar_lci'])
        if 'etstar_uci' in self.products or 'etg_uci' in self.products:
            etstar_uci = ee.Image(etstar_func(evi_sur, etstar_type='uci')) \
                .rename(['etstar_uci'])
        if 'etstar_mean' in self.products:
            output_images.append(etstar_mean)
        if 'etstar_lpi' in self.products:
            output_images.append(etstar_lpi)
        if 'etstar_upi' in self.products:
            output_images.append(etstar_upi)
        if 'etstar_lci' in self.products:
            output_images.append(etstar_lci)
        if 'etstar_uci' in self.products:
            output_images.append(etstar_uci)

        # # For each Landsat scene, I need to calculate water year PPT and ETo sums
        # # ppt = ee.Image.constant(100)
        # # eto = ee.Image.constant(1000)
        # if any([p for p in self.products if 'etg_' in p]):
        #     ppt = ee.Image.constant(refl_toa_orig.get('wy_ppt'))
        #     eto = ee.Image.constant(refl_toa_orig.get('wy_eto'))

        # # ETg
        # if 'etg_mean' in self.products:
        #     etg_mean = ee.Image(etg_func(etstar_mean, eto, ppt)) \
        #         .rename(['etg_mean'])
        #     output_images.append(etg_mean)
        # if 'etg_lpi' in self.products:
        #     etg_lpi = ee.Image(etg_func(etstar_lpi, eto, ppt)) \
        #         .rename(['etg_lpi'])
        #     output_images.append(etg_lpi)
        # if 'etg_upi' in self.products:
        #     etg_upi = ee.Image(etg_func(etstar_upi, eto, ppt)) \
        #         .rename(['etg_upi'])
        #     output_images.append(etg_upi)
        # if 'etg_lci' in self.products:
        #     etg_lci = ee.Image(etg_func(etstar_lci, eto, ppt)) \
        #         .rename(['etg_lci'])
        #     output_images.append(etg_lci)
        # if 'etg_uci' in self.products:
        #     etg_uci = ee.Image(etg_func(etstar_uci, eto, ppt)) \
        #         .rename(['etg_uci'])
        #     output_images.append(etg_uci)

        # Add additional bands
        output_images.extend([
            input_image.select('cloud_score'),
            input_image.select('fmask'),
            input_image.metadata('WRS_ROW', 'row')
        ])

        return ee.Image(output_images) \
            .copyProperties(input_image, system_properties) \
            .set('SCENE_ID', input_image.get('system:index'))

    def tasumi_sr_func(self, refl_toa):
        """Tasumi at-surface reflectance

        Parameters
        ----------
        refl_toa : ee.Image
            Landsat top-of-atmosphere reflectance image.

        Returns
        -------
        ee.Image: at-surface reflectance

        """
        landsat = ee.String(refl_toa.get('system:index')).slice(0, 2)

        scene_date = ee.Date(refl_toa.get('system:time_start'))
        doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
        hour = ee.Number(scene_date.getFraction('day')).multiply(24)
        cos_theta = ee.Image(cos_theta_flat_func(doy, hour))

        pair = ee.Image(pair_func(ee.Image('USGS/SRTMGL1_003')))
        # pair = ee.Image(pair_func(ee.Image('USGS/NED')))

        # Interpolate NLDAS data to scene time
        nldas_image = ee.Image(nldas_interp_func(refl_toa))

        # Specific humidity (kg/kg)
        q = nldas_image.select(['specific_humidity'])
        # q = ee.Image(refl_toa.get('match')).select(['specific_humidity'])
        # q = nldas_interp_func(refl_toa).select(['specific_humidity'])
        ea = pair.expression(
            'q * pair / (0.622 + 0.378 * q)', {'q': q, 'pair': pair})

        # Precipitable water?
        w = pair.multiply(0.14).multiply(ea).add(2.1)

        # Lookup coefficients based on system:index property
        # Need to make sure this hasn't been modified form a merge
        # Could also add a split on the -2 "_"
        c1_dict = ee.Dictionary({
            'LT': [0.987, 2.319, 0.951, 0.375, 0.234, 0.365],
            'LE': [0.987, 2.319, 0.951, 0.375, 0.234, 0.365],
            'LC': [0.987, 2.148, 0.942, 0.248, 0.260, 0.315]})
        c2_dict = ee.Dictionary({
            'LT': [-0.00071, -0.000164, -0.000329, -0.000479, -0.001012, -0.000966],
            'LE': [-0.00071, -0.000164, -0.000329, -0.000479, -0.001012, -0.000966],
            'LC': [-0.000727, -0.000199, -0.000261, -0.000410, -0.001084, -0.000975]})
        c3_dict = ee.Dictionary({
            'LT': [0.000036, 0.000105, 0.00028, 0.005018, 0.004336, 0.004296],
            'LE': [0.000036, 0.000105, 0.00028, 0.005018, 0.004336, 0.004296],
            'LC': [0.000037, 0.000058, 0.000406, 0.000563, 0.000675, 0.004012]})
        c4_dict = ee.Dictionary({
            'LT': [0.088, 0.0437, 0.0875, 0.1355, 0.056, 0.0155],
            'LE': [0.088, 0.0437, 0.0875, 0.1355, 0.056, 0.0155],
            'LC': [0.0869, 0.0464, 0.0928, 0.2256, 0.0632, 0.0116]})
        c5_dict = ee.Dictionary({
            'LT': [0.0789, -1.2697, 0.1014, 0.6621, 0.7757, 0.639],
            'LE': [0.0789, -1.2697, 0.1014, 0.6621, 0.7757, 0.639],
            'LC': [0.0788, -1.0962, 0.1125, 0.7991, 0.7549, 0.6906]})
        cb_dict = ee.Dictionary({
            'LT': [0.640, 0.31, 0.286, 0.189, 0.274, -0.186],
            'LE': [0.640, 0.31, 0.286, 0.189, 0.274, -0.186],
            'LC': [0.640, 0.310, 0.286, 0.189, 0.274, -0.186]})
        c1 = ee.Image.constant(ee.List(c1_dict.get(landsat)))
        c2 = ee.Image.constant(ee.List(c2_dict.get(landsat)))
        c3 = ee.Image.constant(ee.List(c3_dict.get(landsat)))
        c4 = ee.Image.constant(ee.List(c4_dict.get(landsat)))
        c5 = ee.Image.constant(ee.List(c5_dict.get(landsat)))
        cb = ee.Image.constant(ee.List(cb_dict.get(landsat)))

        # Incoming/outgoing narrowband transmittance
        # IN  (C1*exp(((C2*pair)/(Kt*cos_theta))-((C3*W+C4)/cos_theta))+C5)
        # OUT (C1*exp(((C2*pair)/(Kt*1.0))-((C3*W+C4)/1.0))+C5)
        # These broke when I made them expressions, need to try again
        tau_in = pair.multiply(c2).subtract(w.multiply(c3)).subtract(c4) \
            .divide(cos_theta).exp().multiply(c1).add(c5)
        tau_out = pair.multiply(c2).subtract(w.multiply(c3)).subtract(c4) \
            .exp().multiply(c1).add(c5)

        # refl_toa bands should have already been renamed to "_sur" band names
        #   before this function is called
        # SR bands needs to be un-scaled by 0.0001 (from refl_sur_band_func)
        refl_sur = ee.Image(refl_toa).select(refl_sur_bands) \
            .multiply(10000.0) \
            .expression(
                '(b() + cb * (tau_in - 1.0)) / (tau_in * tau_out)',
                {'cb': cb, 'tau_in': tau_in, 'tau_out': tau_out}) \
            .rename(refl_sur_bands)

        return refl_sur \
            .clamp(0.0001, 1) \
            .copyProperties(refl_toa, system_properties)

    def tasumi_sr_adjust_func(self, refl_sur):
        """Apply cross sensor calibration adjustments to Tasumi SR images

        Parameters
        ----------
        refl_sur : ee.Image
            Landsat at-surface reflectance image.

        Returns
        -------
        ee.Image

        Notes
        -----
        http://www.sciencedirect.com/science/article/pii/S0034425716302619
        Coefficients are for scaling OLI to ETM+
        Invert the calculation for ETM+ to OLI

        """
        landsat = ee.String(refl_sur.get('system:index')).slice(0, 2)

        if not self.adjust_method:
            return refl_sur
        elif self.adjust_method.lower() == 'etm_2_oli':
            m_dict = ee.Dictionary({
                'LT': [1.0, 1.0, 1.0047, 1.0036, 1.0, 1.0],
                'LE': [1.0, 1.0, 1.0047, 1.0036, 1.0, 1.0],
                'LC': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
            b_dict = ee.Dictionary({
                'LT': [0.0, 0.0, 0.0024, -0.0003, 0.0, 0.0],
                'LE': [0.0, 0.0, 0.0024, -0.0003, 0.0, 0.0],
                'LC': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
            m = ee.Image.constant(ee.List(m_dict.get(landsat)))
            b = ee.Image.constant(ee.List(b_dict.get(landsat)))
            return ee.Image(refl_sur).subtract(b).divide(m)
        elif self.adjust_method.lower() == 'oli_2_etm':
            m_dict = ee.Dictionary({
                'LT': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                'LE': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                'LC': [1.0, 1.0, 1.0047, 1.0036, 1.0, 1.0]})
            b_dict = ee.Dictionary({
                'LT': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'LE': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'LC': [0.0, 0.0, 0.0024, -0.0003, 0.0, 0.0]})
            m = ee.Image.constant(ee.List(m_dict.get(landsat)))
            b = ee.Image.constant(ee.List(b_dict.get(landsat)))
            return ee.Image(refl_sur).multiply(m).add(b)

    def usgs_sr_adjust_func(self, refl_sur):
        """Apply cross sensor calibration adjustments to USGS SR images

        Parameters
        ----------
        refl_sur : ee.Image
            Landsat at-surface reflectance image.

        Returns
        -------
        ee.Image

        Notes
        -----
        http://www.sciencedirect.com/science/article/pii/S0034425716302619
        Coefficients are for scaling OLI to ETM+
        Invert the calculation for ETM+ to OLI

        """
        landsat = ee.String(refl_sur.get('system:index')).slice(0, 2)

        if not self.adjust_method:
            return refl_sur
        elif self.adjust_method.lower() == 'etm_2_oli':
            m_dict = ee.Dictionary({
                'LT': [1.0, 1.0, 0.9911, 0.9892, 1.0, 1.0],
                'LE': [1.0, 1.0, 0.9911, 0.9892, 1.0, 1.0],
                'LC': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
            b_dict = ee.Dictionary({
                'LT': [0.0, 0.0, 0.0099, 0.007, 0.0, 0.0],
                'LE': [0.0, 0.0, 0.0099, 0.007, 0.0, 0.0],
                'LC': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
            m = ee.Image.constant(ee.List(m_dict.get(landsat)))
            b = ee.Image.constant(ee.List(b_dict.get(landsat)))
            return ee.Image(refl_sur).subtract(b).divide(m)
        elif self.adjust_method.lower() == 'oli_2_etm':
            m_dict = ee.Dictionary({
                'LT': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                'LE': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                'LC': [1.0, 1.0, 0.9911, 0.9892, 1.0, 1.0]})
            b_dict = ee.Dictionary({
                'LT': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'LE': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'LC': [0.0, 0.0, 0.0099, 0.007, 0.0, 0.0]})
            m = ee.Image.constant(ee.List(m_dict.get(landsat)))
            b = ee.Image.constant(ee.List(b_dict.get(landsat)))
            return ee.Image(refl_sur).multiply(m).add(b)


# def scene_id_func(img):
#     """Construct Collecton 1 short SCENE_ID for collection 1 images
#
#     LT05_PPPRRR_YYYYMMDD
#     Format matches EE collection 1 system:index
#     Split on '_' in case the collection was merged first
#     """
#     scene_id = ee.List(ee.String(
#         img.get('system:index')).split('_')).slice(-3)
#     scene_id = ee.String(scene_id.get(0)).cat('_') \
#         .cat(ee.String(scene_id.get(1))).cat('_') \
#         .cat(ee.String(scene_id.get(2)))
#     return img.setMulti({'SCENE_ID': scene_id})


def cos_theta_flat_func(acq_doy, acq_time, lat=None, lon=None):
    """Cos(theta) - Spatially varying flat Model

    Args:
        acq_doy (ee.Number): Image acquisition day of year.
            scene_date = ee.Date(ee_image.get('system:time_start'))
            acq_doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
        acq_time (ee.Number): Image acquisition UTC time in hours.
            i.e. 18:30 -> 18.5
            scene_date = ee.Date(ee_image.get('system:time_start'))
            acq_time = ee.Number(scene_date.getFraction('day')).multiply(24)
        lat (ee.Image): Latitude [radians].
            lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi/180)
        lon (ee.Image): Longitude [radians].
            lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi/180)

    Returns:
        ee.Image()
    """
    pi = math.pi
    if lat is None:
        lat = ee.Image.pixelLonLat() \
            .select(['latitude']).multiply(pi / 180)
    if lon is None:
        lon = ee.Image.pixelLonLat() \
            .select(['longitude']).multiply(pi / 180)
    delta = acq_doy.multiply(2 * pi / 365).subtract(1.39435) \
        .sin().multiply(0.40928)
    sc_b = acq_doy.subtract(81).multiply(2 * pi / 364)
    sc = sc_b.multiply(2).sin().multiply(0.1645) \
        .subtract(sc_b.cos().multiply(0.1255)) \
        .subtract(sc_b.sin().multiply(0.025))
    solar_time = lon.multiply(12 / pi).add(acq_time).add(sc)
    # solar_time = lon.expression(
    #     't + (lon * 12 / pi) + sc',
    #     {'pi':pi, 't':ee.Image.constant(acq_time),
    #      'lon':lon, 'sc':ee.Image.constant(sc)})
    omega = solar_time.subtract(12).multiply(pi / 12)
    cos_theta = lat.expression(
        'sin(delta) * sin(lat) + cos(delta) * cos(lat) * cos(omega)',
        {'delta': ee.Image.constant(delta), 'lat': lat, 'omega': omega})
    return cos_theta.select([0], ['cos_theta'])


def cos_theta_mountain_func(acq_doy, acq_time, lat=None, lon=None,
                            slope=None, aspect=None):
    """Cos(theta) - Spatially varying moutain model

    Args:
        acq_doy: EarthEngine number of the image acquisition day of year
            scene_date = ee.Algorithms.Date(ee_image.get('system:time_start'))
            acq_doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
        acq_time: EarthEngine number of the image acquisition UTC time in hours
            i.e. 18:30 -> 18.5
            Calcuatl
            scene_date = ee.Algorithms.Date(ee_image.get('system:time_start'))
            acq_time = ee.Number(scene_date.getFraction('day')).multiply(24)
        lat: EarthEngine image of the latitude [radians]
            lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi/180)
        lon: EarthEngine image of the longitude [radians]
            lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi/180)
        slope: EarthEngine image of the slope [radians]
            terrain = ee.call('Terrain', ee.Image("USGS/NED"))
            slope = terrain.select(["slope"]).multiply(pi/180)
        aspect: EarthEngine image of the aspect [radians]
            0 is south, so subtract Pi from traditional aspect raster/calc
            terrain = ee.call('Terrain', ee.Image("USGS/NED"))
            aspect = terrain.select(["aspect"]).multiply(pi/180).subtract(math.pi)

    Returns:
        ee.Image()
    """
    pi = math.pi
    if lat is None:
        lat = ee.Image.pixelLonLat() \
            .select(['latitude']).multiply(pi / 180)
    if lon is None:
        lon = ee.Image.pixelLonLat() \
            .select(['longitude']).multiply(pi / 180)
    if slope is None or aspect is None:
        terrain = ee.call('Terrain', ee.Image('USGS/NED'))
    if slope is None:
        slope = terrain.select(['slope']).multiply(pi / 180)
    if aspect is None:
        aspect = terrain.select(['aspect']).multiply(pi / 180).subtract(pi)
    delta = acq_doy.multiply(2 * math.pi / 365).subtract(1.39435) \
        .sin().multiply(0.40928)
    b = acq_doy.subtract(81).multiply(2 * pi / 364)
    sc = b.multiply(2).sin().multiply(0.1645)\
        .subtract(b.cos().multiply(0.1255))\
        .subtract(b.sin().multiply(0.025))
    solar_time = lon.multiply(12 / pi).add(acq_time).add(sc)
    # solar_time = lon.expression(
    #   't + (lon * 12 / pi) + sc',
    #   {'pi':pi, 't':ee.Image.constant(acq_time),
    #    'lon':lon, 'sc':ee.Image.constant(sc)})
    omega = solar_time.subtract(12).multiply(pi / 12)
    slope_c = slope.cos()
    slope_s = slope.sin()
    cos_theta = lat.expression(
        '(sin(lat) * slope_c * delta_s) - '
        '(cos(lat) * slope_s * cos(aspect) * delta_s) + '
        '(cos(lat) * slope_c * cos(omega) * delta_c) + '
        '(sin(lat) * slope_s * cos(aspect) * cos(omega) * delta_c) + '
        '(sin(aspect) * slope_s * sin(omega) * delta_c)',
        {'lat': lat, 'aspect': aspect,
         'slope_c': slope_c, 'slope_s': slope_s, 'omega': omega,
         'delta_c': ee.Image.constant(delta.cos()),
         'delta_s': ee.Image.constant(delta.sin())})
    cos_theta = cos_theta.divide(slope_c).max(ee.Image.constant(0.1))

    return cos_theta.rename(['cos_theta'])


def landsat_albedo_func(refl_img):
    """Albedo"""
    wb_coef = [0.254, 0.149, 0.147, 0.311, 0.103, 0.036]
    return ee.Image(refl_img).multiply(wb_coef).reduce(ee.Reducer.sum())


def landsat_ndvi_func(refl_img, bands):
    """Calculate NDVI for a daily Landsat 4, 5, 7, or 8 image"""
    # Removed .clamp(-0.1, 1)
    return ee.Image(refl_img).normalizedDifference(['nir', 'red'])


# def landsat_savi_func(refl_image, L=0.1):
#     """Soil adjusted vegetation index (SAVI)"""
#     return ee.Image(refl_img).expression(
#         '(1.0 + L) * (nir - red) / (L + nir + red)',
#         {'red': refl_image.select('red'),
#          'nir': refl_image.select('nir'), 'L': L})


# def savi_func(refl_image, L=0.1):
#     """Soil adjusted vegetation index (SAVI)"""
#     return ee.Image(refl_img).expression(
#         '(1.0 + L) * (nir - red) / (L + nir + red)',
#         {'red': refl_image.select('red'),
#          'nir': refl_image.select('nir'), 'L': L})


def savi_lai_func(savi):
    """Leaf area index (LAI) calculated from SAVI"""
    return savi.pow(3).multiply(11.0).clamp(0, 6)


def ndvi_lai_func(ndvi):
    """Leaf area index (LAI) calculated from NDVI"""
    return ndvi.pow(3).multiply(7.0).clamp(0, 6)


def landsat_evi_func(refl_image):
    """Calculate EVI for a daily Landsat 4, 5, 7, or 8 image"""
    return ee.Image(refl_image).expression(
        '(2.5 * (NIR - RED)) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0)',
        {'BLUE': refl_image.select('blue'),
         'RED': refl_image.select('red'),
         'NIR': refl_image.select('nir')})


def landsat_savi_func(refl_image):
    """Calculate SAVI for a daily Landsat 4, 5, 7, or 8 image"""
    return ee.Image(refl_image) \
        .expression(
            '((NIR - RED) / (NIR + RED + 0.5)) * (1.0 + 0.5)',
            {'RED': refl_image.select('red'),
             'NIR': refl_image.select('nir')})


def landsat_msavi_func(refl_image):
    """Calculate MSAVI for a daily Landsat 4, 5, 7, or 8 image"""
    return ee.Image(refl_image) \
        .expression(
            '(2 * NIR + 1 - sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))) / 2.0',
            {'RED': refl_image.select('red'),
             'NIR': refl_image.select('nir')})


def beamer_func(img):
    """Compute Beamer ET*/ET/ETg directly from EVI

    Args:
        img (ee.Image): Image with "evi_sur" band.
            Must have properties of "wy_eto" and "wy_ppt"

    Returns:
        ee.Image() of ETg
    """
    # EVI
    #evi_sur = ee.Image(img).select(['evi_sur'])
    ndvi_sur = ee.Image(img).select(['ndvi_sur'])
    #savi_sur = ee.Image(img).select(['savi_sur'])

    # ET*
    etstar_mean = etstar_func(ndvi_sur, 'mean').rename(['etstar_mean'])
    etstar_lpi = etstar_func(ndvi_sur, 'lpi').rename(['etstar_lpi'])
    etstar_upi = etstar_func(ndvi_sur, 'upi').rename(['etstar_upi'])
    etstar_lci = etstar_func(ndvi_sur, 'lci').rename(['etstar_lci'])
    etstar_uci = etstar_func(ndvi_sur, 'uci').rename(['etstar_uci'])

    # For each Landsat scene, I need to calculate water year PPT and ETo sums
    ppt = ee.Image.constant(ee.Number(img.get('wy_ppt')))
    eto = ee.Image.constant(ee.Number(img.get('wy_eto')))

    # ETg
    etg_mean = etg_func(etstar_mean, eto, ppt).rename(['etg_mean'])
    etg_lpi = etg_func(etstar_lpi, eto, ppt).rename(['etg_lpi'])
    etg_upi = etg_func(etstar_upi, eto, ppt).rename(['etg_upi'])
    etg_lci = etg_func(etstar_lci, eto, ppt).rename(['etg_lci'])
    etg_uci = etg_func(etstar_uci, eto, ppt).rename(['etg_uci'])

    # ET
    et_mean = et_func(etg_mean, ppt).rename(['et_mean'])
    et_lpi = et_func(etg_lpi, ppt).rename(['et_lpi'])
    et_upi = et_func(etg_upi, ppt).rename(['et_upi'])
    et_lci = et_func(etg_lci, ppt).rename(['et_lci'])
    et_uci = et_func(etg_uci, ppt).rename(['et_uci'])

    return ee.Image([etg_mean, etg_lpi, etg_upi, etg_lci, etg_uci]) \
        .copyProperties(img, ['system:index', 'system:time_start'])


#def etstar_func(evi, etstar_type='mean', evi_min=0.0700019):
def etstar_func(ndvi, etstar_type='mean', ndvi_min=0.0315):
#def etstar_func(savi, etstar_type='mean', savi_min=0.035):
    """Compute Beamer ET* from EVI (assuming at-surface reflectance)"""
    #def etstar(img, c0, c1, c2):
    def etstar(img, c0, c1):       
        """Beamer ET*"""
        return ee.Image(img) \
            .max(ndvi_min) \
            .expression(
                #'c0 + c1 * b(0) + c2 * (b(0) ** 2)',
                'c0 + c1 * b(0)',
                #{'c0': c0, 'c1': c1, 'c2': c2}) \
            #.max(0)    
                {'c0': c0, 'c1': c1}) \
            .max(0)
    if etstar_type == 'mean':
        #return etstar(evi, -0.1955, 2.9042, -1.5916)
        return etstar(ndvi, -0.035, 1.1277)
        #return etstar(savi, -0.0983, 3.2848, -2.2715)
    elif etstar_type == 'lpi':
        #return etstar(evi, -0.2871, 2.9192, -1.6263)
        return etstar(ndvi, -0.1667, 1.1118)
        #return etstar(savi, -0.2734, 3.3057, -2.347)
    elif etstar_type == 'upi':
        #return etstar(evi, -0.1039, 2.8893, -1.5569)
        return etstar(ndvi, 0.0967, 1.1437)
        #return etstar(savi, 0.0767, 3.2639, -2.1959)
    elif etstar_type == 'lci':
        #return etstar(evi, -0.2142, 2.9175, -1.6554)
        return etstar(ndvi, -0.0469, 1.0733)
        #return etstar(savi, -0.1244, 3.3001, -2.4427)
    elif etstar_type == 'uci':
        #return etstar(evi, -0.1768, 2.8910, -1.5278)
        return etstar(ndvi, -0.0231, 1.1822)
        #return etstar(savi, -0.0722, 3.2695, -2.1003)


def etg_func(etstar, eto, ppt):
    """Compute groundwater ET (ETg) (ET* x (ETo - PPT))"""
    return etstar.multiply(eto.subtract(ppt))


def et_func(etg, ppt):
    """Compute net ET (ETg + PPT)"""
    return etg.add(ppt)


# def tasseled_cap_func(self, refl_toa):
#     refl_toa_sub = refl_toa.select(refl_bands)
#     tc_bright_coef = ee.List(refl_toa.get('tc_bright'))
#     tc_green_coef = ee.List(refl_toa.get('tc_green'))
#     tc_wet_coef = ee.List(refl_toa.get('tc_wet'))
#     return ee.Image([
#         refl_toa_sub.multiply(tc_bright_coef).reduce(ee.Reducer.sum()),
#         refl_toa_sub.multiply(tc_green_coef).reduce(ee.Reducer.sum()),
#         refl_toa_sub.multiply(tc_wet_coef).reduce(ee.Reducer.sum())])\
#         .select([0, 1, 2], ['tc_bright', 'tc_green', 'tc_wet'])


def tc_bright_func(refl_toa):
    """Tasseled cap brightness

    Top of atmosphere (at-satellite) reflectance

    LT04/LT05 - http://www.gis.usu.edu/~doug/RS5750/assign/OLD/RSE(17)-301.pdf
    LE07 - http://landcover.usgs.gov/pdf/tasseled.pdf
    LC08 - http://www.tandfonline.com/doi/abs/10.1080/2150704X.2014.915434
    https://www.researchgate.net/publication/262005316_Derivation_of_a_tasselled_cap_transformation_based_on_Landsat_8_at_satellite_reflectance
    """
    landsat = ee.String(refl_toa.get('system:index')).slice(0, 2)
    bright_coef = ee.Dictionary({
        'LT': [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303],
        'LE': [0.3561, 0.3972, 0.3904, 0.6966, 0.2286, 0.1596],
        'LC': [0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872],
    })
    return refl_toa.multiply(ee.Image.constant(ee.List(bright_coef.get(landsat)))) \
        .reduce(ee.Reducer.sum()) \
        .rename(['tc_bright'])


def tc_green_func(refl_toa):
    """Tasseled cap greeness"""
    landsat = ee.String(refl_toa.get('system:index')).slice(0, 2)
    green_coef = ee.Dictionary({
        'LT': [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446],
        'LE': [-0.3344, -0.3544, -0.4556, 0.6966, -0.0242, -0.2630],
        'LC': [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608],
    })
    return refl_toa.multiply(ee.Image.constant(ee.List(green_coef.get(landsat)))) \
        .reduce(ee.Reducer.sum()) \
        .rename(['tc_green'])


def tc_wet_func(refl_toa):
    """Tasseled cap wetness"""
    landsat = ee.String(refl_toa.get('system:index')).slice(0, 2)
    wet_coef = ee.Dictionary({
        'LT': [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109],
        'LE': [0.2626, 0.2141, 0.0926, 0.0656, -0.7629, -0.5388],
        'LC': [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
    })
    return refl_toa.multiply(ee.Image.constant(ee.List(wet_coef.get(landsat)))) \
        .reduce(ee.Reducer.sum()) \
        .rename(['tc_wet'])


def em_nb_func(ndvi, lai):
    """Narrowband emissivity"""
    # Initial values are for NDVI > 0 and LAI <= 3
    return lai.divide(300).add(0.97) \
        .where(ndvi.lte(0), 0.99) \
        .where(ndvi.gt(0).And(lai.gt(3)), 0.98)


def em_wb_func(ndvi, lai):
    """Broadband emissivity"""
    # Initial values are for NDVI > 0 and LAI <= 3
    return lai.divide(100).add(0.95) \
        .where(ndvi.lte(0), 0.985) \
        .where(ndvi.gt(0).And(lai.gt(3)), 0.98)


def ts_func(ts_brightness, em_nb, k1=607.76, k2=1260.56):
    """Surface temperature"""
    # First back out radiance from brightness temperature
    # Then recalculate emissivity corrected Ts
    thermal_rad_toa = ts_brightness.expression(
        'k1 / (exp(k2 / ts_brightness) - 1.0)',
        {'ts_brightness': ts_brightness, 'k1': k1, 'k2': k2})
    rc = thermal_rad_toa.expression(
        '((thermal_rad_toa - rp) / tnb) - ((1.0 - em_nb) * rsky)',
        {"thermal_rad_toa": thermal_rad_toa, "em_nb": em_nb,
         "rp": 0.91, "tnb": 0.866, 'rsky': 1.32})
    ts = rc.expression(
        'k2 / log(em_nb * k1 / rc + 1.0)',
        {'em_nb': em_nb, 'rc': rc, 'k1': k1, "k2": k2})
    return ts.rename(['ts'])


# def landsat_true_color_func(img):
#     """Calculate true color for a daily Landsat 4, 5, 7, or 8 image"""
#     return ee.Image(img.select(['blue', 'green', 'red']))\
#         .copyProperties(img, system_properties)


# def landsat_false_color_func(img):
#     """Calculate false color for a daily Landsat 4, 5, 7, or 8 image"""
#     return ee.Image(img.select(['green', 'red', 'nir']))\
#         .copyProperties(img, system_properties)


def nldas_interp_func(img):
    """Interpolate NLDAS image at Landsat scene time

    Parameters
    ----------
    img : ee.Image

    Returns
    -------
    ee.Image : NLDAS values interpolated at the image time

    """
    scene_time = ee.Number(img.get('system:time_start'))
    scene_datetime = ee.Date(scene_time)
    nldas_coll = ee.ImageCollection('NASA/NLDAS/FORA0125_H002')
    nldas_prev_image = ee.Image(nldas_coll.filterDate(
        scene_datetime.advance(-1, 'hour'), scene_datetime).first())
    nldas_next_image = ee.Image(nldas_coll.filterDate(
        scene_datetime, scene_datetime.advance(1, 'hour')).first())
    nldas_prev_time = ee.Number(nldas_prev_image.get('system:time_start'))
    nldas_next_time = ee.Number(nldas_next_image.get('system:time_start'))

    # Calculate time ratio of Landsat image between NLDAS images
    time_ratio = scene_time.subtract(nldas_prev_time).divide(
        nldas_next_time.subtract(nldas_prev_time))
    # time_ratio_image = ee.Image.constant(scene_time.subtract(nldas_prev_time) \
    #     .divide(nldas_next_time.subtract(nldas_prev_time)))

    # Interpolate NLDAS values at Landsat image time
    return nldas_next_image.subtract(nldas_prev_image) \
        .multiply(time_ratio).add(nldas_prev_image) \
        .setMulti({'system:time_start': scene_time})


def landsat_acca_band_func(refl_toa_img):
    """Add ACCA like cloud score band to Landsat collection"""
    cloud_score = ee.Algorithms.Landsat.simpleCloudScore(refl_toa_img) \
        .select(['cloud'], ['cloud_score'])
    return refl_toa_img.addBands(cloud_score)


def landsat_bqa_fmask_func(refl_toa_img): # Mapped to TOA coll 7 Mar 2022 (line 378)
    """Extract Fmask image from Landsat TOA Collection 1 QA band

    https://landsat.usgs.gov/collectionqualityband
    https://code.earthengine.google.com/356a3580096cca315785d0859459abbd

    Confidence values
    00 = "Not Determined" = Algorithm did not determine the status of this condition
    01 = "No" = Algorithm has low to no confidence that this condition exists
        (0-33 percent confidence)
    10 = "Maybe" = Algorithm has medium confidence that this condition exists
        (34-66 percent confidence)
    11 = "Yes" = Algorithm has high confidence that this condition exists
        (67-100 percent confidence

    Set fmask band to old style Fmask values
        0 - Clear land
        1 - Clear water (not available in BQA)
        2 - Cloud shadow
        3 - Snow
        4 - Cloud (with L8 cirrus)
    """
    qa_img = ee.Image(refl_toa_img.select(['BQA']))

    # # Extracting cloud masks from BQA using rightShift() and  bitwiseAnd()
    # # Cloud (med & high confidence), then snow, then shadow, then fill
    # # Low confidence clouds tend to be the FMask buffer
    # # Bits 11/12 will pull cirrus mask from Landsat 8
    fill_mask = qa_img.bitwiseAnd(1).neq(0)                        # bits: 0
    # drop_mask = qa_img.rightShift(1).bitwiseAnd(2).neq(0);       # bits: 1
    # saturation_mask = qa_img.rightShift(3).bitwiseAnd(3).gte(8)  # bits: 2, 3
    cloud_mask = qa_img.rightShift(4).bitwiseAnd(1).neq(0)         # bits: 4
    cloud_conf = qa_img.rightShift(5).bitwiseAnd(3).gte(2)         # bits: 5, 6
    shadow_mask = qa_img.rightShift(7).bitwiseAnd(3).gte(3)        # bits: 7, 8
    snow_mask = qa_img.rightShift(9).bitwiseAnd(3).gte(3)          # bits: 9, 10
    cirrus_mask = qa_img.rightShift(11).bitwiseAnd(3).gte(3)       # bits: 11 12
    fmask_img = fill_mask \
        .add(shadow_mask.multiply(2)) \
        .add(snow_mask.multiply(3)) \
        .add(cloud_mask.And(cloud_conf).Or(cirrus_mask).multiply(4))

    return refl_toa_img.addBands(fmask_img.rename(['fmask']))


# def landsat_pixel_qa_fmask_func(refl_sur_img): # NOT used in script as of 7 Mar 2022
#     """Extract Fmask image from Landsat SR Collection 1 pixel_qa band

#     https://landsat.usgs.gov/sites/default/files/documents/ledaps_product_guide.pdf
#     https://code.earthengine.google.com/eb6ce4a7af177670a6038c4bd53724fe

#     Set fmask band to old style Fmask values
#         0 - Clear land
#         1 - Clear water
#         2 - Cloud shadow
#         3 - Snow
#         4 - Cloud
#     """
#     qa_img = ee.Image(refl_sur_img.select(['pixel_qa']))

#     # Extracting cloud masks from pixel_qa using rightShift() and bitwiseAnd()
#     # Cloud (med & high confidence), then snow, then shadow, then fill
#     # Low confidence clouds tend to be the FMask buffer
#     fill_mask = qa_img.bitwiseAnd(1).neq(0)                   # bits: 0
#     # clear_mask = qa_img.rightShift(1).bitwiseAnd(2).neq(0)  # bits: 1
#     water_mask = qa_img.rightShift(2).bitwiseAnd(1).neq(0)    # bits: 2
#     shadow_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)   # bits: 3
#     snow_mask = qa_img.rightShift(4).bitwiseAnd(1).neq(0)     # bits: 4
#     cloud_mask = qa_img.rightShift(5).bitwiseAnd(1).neq(0)    # bits: 5
#     cloud_conf = qa_img.rightShift(6).bitwiseAnd(3).gte(1)    # bits: 6, 7
#     fmask_img = fill_mask \
#         .add(water_mask.multiply(1)) \
#         .add(shadow_mask.multiply(2)) \
#         .add(snow_mask.multiply(3)) \
#         .add(cloud_mask.And(cloud_conf).multiply(4))

#     return refl_sur_img.addBands(fmask_img.rename(['fmask']))


# Cloud Mask for col 2 SR using bits (GS March 2022)
#  returns 'fmask' band to represent
def landsat_col2_cloudmask_func(col2_img):
    cirrus_flag = True
    dilate_flag = True
    qa_img = col2_img.select(['QA_PIXEL'])
    #     .And(qa_img.rightShift(6).bitwiseAnd(3).gte(cloud_confidence)) issue with bit 6
    water_mask = qa_img.rightShift(7).bitwiseAnd(1).neq(0)
    shadow_mask = qa_img.rightShift(4).bitwiseAnd(1).neq(0)
    snow_mask = qa_img.rightShift(5).bitwiseAnd(1).neq(0)
    cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
    if cirrus_flag:
        cloud_mask = cloud_mask.Or(qa_img.rightShift(2).bitwiseAnd(1).neq(0))
    if dilate_flag:
        cloud_mask = cloud_mask.Or(qa_img.rightShift(1).bitwiseAnd(1).neq(0))

    fmask_img = water_mask \
        .add(shadow_mask.multiply(2)) \
        .add(snow_mask.multiply(3)) \
        .add(cloud_mask.multiply(4))

    return col2_img.addBands(fmask_img.rename(['fmask']))



    # # Extracting cloud masks from qa band using band values directly
    # # Cloud, then shadow, then snow, then fill
    # cloud_mask = qa_img.eq(224).Or(qa_img.eq(160)) \
    #     .Or(qa_img.eq(72).Or(qa_img.eq(136))) \
    #     .Or(qa_img.eq(80).Or(qa_img.eq(112)).Or(qa_img.eq(144)).Or(qa_img.eq(176))) \
    #     .Or(qa_img.eq(1))


# def landsat_qa_band_func(refl_img):
#     """Get Fmask band from the joined properties
#
#     https://landsat.usgs.gov/collectionqualityband
#
#     Confidence values
#     00 = "Not Determined" = Algorithm did not determine the status of this condition
#     01 = "No" = Algorithm has low to no confidence that this condition exists (0-33 percent confidence)
#     10 = "Maybe" = Algorithm has medium confidence that this condition exists (34-66 percent confidence)
#     11 = "Yes" = Algorithm has high confidence that this condition exists (67-100 percent confidence
#     """
#     qa_img = ee.Image(refl_img.select(['BQA']))
#
#     def getQABits(image, start, end, newName):
#         """
#         Tyler's function from https://ee-api.appspot.com/#97ab9a8f694b28128a5a5ca2e2df7841
#         """
#         pattern = 0
#         for i in range(start, end + 1):
#             pattern += int(2 ** i)
#         return image.select([0], [newName]) \
#             .bitwise_and(pattern).right_shift(start)
#
#     # Extract the various masks from the QA band
#     fill_mask = getQABits(qa_img, 0, 0, 'designated_fill')
#     # drop_mask = getQABits(qa_img, 1, 1, 'dropped_pixel')
#     # Landsat 8 only
#     # terrain_mask = getQABits(qa_img, 1, 1, 'terrain_occlusion')
#     # saturation_mask = getQABits(qa_img, 2, 3, 'saturation_confidence').gte(2)
#     # cloud_mask = getQABits(qa_img, 4, 4, 'cloud')
#     cloud_mask = getQABits(qa_img, 5, 6, 'cloud_confidence').gte(2)
#     shadow_mask = getQABits(qa_img, 7, 8, 'shadow_confidence').gte(3)
#     snow_mask = getQABits(qa_img, 9, 10, 'snow_confidence').gte(3)
#     # Landsat 8 only
#     # cirrus_mask = getQABits(qa_img, 11, 12, 'cirrus_confidence').gte(3)
#
#     # Convert masks to old style Fmask values
#     # 0 - Clear land
#     # 1 - Clear water
#     # 2 - Cloud shadow
#     # 3 - Snow
#     # 4 - Cloud
#     fmask_img = fill_mask \
#         .add(shadow_mask.multiply(2)) \
#         .add(snow_mask.multiply(3)) \
#         .add(cloud_mask.multiply(4))
#
#     return refl_img.addBands(fmask_img.rename(['fmask']))


def landsat_sur_band_func(refl_img):
    """Get at-surface reflectance bands from the joined properties"""
    return refl_img \
        .addBands(ee.Image(refl_img.get('sur')).rename(refl_sur_bands))


def landsat5_toa_band_func(refl_img):
    """Rename Landsat 4 and 5 bands to common band names

    Change band order to match Landsat 8
    Set K1 and K2 coefficients used for computing land surface temperature
    Set Tasseled cap coefficients
    """
    return refl_img \
        .select(
            ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'cloud_score', 'fmask'],
            refl_toa_bands + ['lst', 'cloud_score', 'fmask']) \
        .copyProperties(refl_img, system_properties) \
        .setMulti({
            'k1_constant': refl_img.get('K1_CONSTANT_BAND_6'),
            'k2_constant': refl_img.get('K2_CONSTANT_BAND_6')})


def landsat7_toa_band_func(refl_img):
    """Rename Landsat 7 bands to common band names

    For now, don't include pan-chromatic or high gain thermal band
    Change band order to match Landsat 8
    Set K1 and K2 coefficients used for computing land surface temperature
    Set Tasseled cap coefficients

    ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2',
     'B7', 'B8', 'cloud_score', 'fmask'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'thermal1', 'thermal2',
     'swir2', 'pan', 'cloud_score', 'fmask'])
    """
    return refl_img \
        .select(
            ['B1', 'B2', 'B3', 'B4', 'B5', 'B7',
             'B6_VCID_1', 'cloud_score', 'fmask'],  
            refl_toa_bands + ['lst', 'cloud_score', 'fmask']) \
        .copyProperties(refl_img, system_properties) \
        .setMulti({
            'k1_constant': refl_img.get('K1_CONSTANT_BAND_6_VCID_1'),
            'k2_constant': refl_img.get('K2_CONSTANT_BAND_6_VCID_1')})


def landsat8_toa_band_func(refl_img):
    """Rename Landsat 8 bands to common band names

    For now, don't include coastal, cirrus, pan-chromatic, or 2nd thermal band
    Set K1 and K2 coefficients used for computing land surface temperature
    Set Tasseled cap coefficients

    ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
     'B9', 'B10', 'B11', 'cloud_score'],
    ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
     'pan', 'cirrus', 'thermal1', 'thermal2', 'cloud_score'])
    """
    return refl_img \
        .select(
            ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
             'B10', 'cloud_score', 'fmask'],
            refl_toa_bands + ['lst', 'cloud_score', 'fmask']) \
        .copyProperties(refl_img, system_properties) \
        .setMulti({
            'k1_constant': refl_img.get('K1_CONSTANT_BAND_10'),
            'k2_constant': refl_img.get('K2_CONSTANT_BAND_10')})


def landsat9_toa_band_func(refl_img):
    """Rename Landsat 9 bands to common band names

    For now, don't include coastal, cirrus, pan-chromatic, or 2nd thermal band
    Set K1 and K2 coefficients used for computing land surface temperature
    Set Tasseled cap coefficients

    ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
     'B9', 'B10', 'B11', 'cloud_score'],
    ['coastal', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2',
     'pan', 'cirrus', 'thermal1', 'thermal2', 'cloud_score'])
    """
    return refl_img \
        .select(
            ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
             'B10', 'cloud_score', 'fmask'],
            refl_toa_bands + ['lst', 'cloud_score', 'fmask']) \
        .copyProperties(refl_img, system_properties) \
        .setMulti({
            'k1_constant': refl_img.get('K1_CONSTANT_BAND_10'),
            'k2_constant': refl_img.get('K2_CONSTANT_BAND_10')})


def landsat5_sur_band_func(refl_img):
    """Rename Landsat 4 and 5 bands to common band names

    Change band order to match Landsat 8/9
    Scale values by 0.0001
    """
    return refl_img \
        .select(
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'], 
            refl_sur_bands) \
        .multiply(0.0000275).add(-0.2) \
        .copyProperties(refl_img, system_properties)


def landsat7_sur_band_func(refl_img):
    """Rename Landsat 7 bands to common band names

    Change band order to match Landsat 8/9
    For now, don't include pan-chromatic or high gain thermal band
    Scale values by collection 2 vals (2.75e-05 - 0.2)
    """
    return refl_img \
        .select(
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'], 
            refl_sur_bands) \
        .multiply(0.0000275).add(-0.2) \
        .copyProperties(refl_img, system_properties)


def landsat8_sur_band_func(refl_img):
    """Rename Landsat 8 bands to common band names

    For now, don't include coastal, cirrus, or pan-chromatic
    Scale values by collection 2 vals (2.75e-05 - 0.2)
    """
    return refl_img \
        .select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            refl_sur_bands) \
        .multiply(0.0000275).add(-0.2) \
        .copyProperties(refl_img, system_properties)


def landsat9_sur_band_func(refl_img):
    """Rename Landsat 8 bands to common band names

    For now, don't include coastal, cirrus, or pan-chromatic
    Scale values by collection 2 vals (2.75e-05 - 0.2)
    """
    return refl_img \
        .select(
            ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            refl_sur_bands) \
        .multiply(0.0000275).add(-0.2) \
        .copyProperties(refl_img, system_properties)


# def common_area_func(img):
#     """Only keep pixels that are common to all bands"""
#     common_mask = ee.Image(img).mask().reduce(ee.Reducer.And())
#     return img.updateMask(common_mask)


# def erode_func(img):
#     """"""
#     input_mask = ee.Image(img).mask().reduceNeighborhood(
#         ee.Reducer.min(), ee.call("Kernel.circle", 120, 'meters'))
#     return img.updateMask(input_mask)


def landsat_acca_cloud_mask_func(img):
    """Apply basic ACCA cloud mask to a daily Landsat TOA image

    Only apply ACCA cloud mask to Landsat reflectance bands

    For Landsat 8 images after Oct 31st, 2015, there is no LST data
        so simpleCloudScore returns a fully masked image
    This makes it appear as if there are no Landsat 8 TOA images/data
    If simpleCloudScore doesn't work, this function should not mask any values
        and instead return all pixels, even cloudy ones
    Use "unmask(0)" to set all masked pixels as cloud free
    This should have no impact on earlier Landsat TOA images and could be
        removed once the LST issue is resolved
    """
    cloud_mask = img.select(['cloud_score']).unmask(0).lt(50)
    return img \
        .select(refl_toa_bands + refl_sur_bands + ['lst']) \
        .updateMask(cloud_mask) \
        .addBands(img.select(['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


def landsat_fmask_cloud_mask_func(img):
    """Apply the Fmask band in the reflectance collections

    Only apply Fmask cloud mask to Landsat reflectance bands

    0 - Clear land
    1 - Clear water
    2 - Cloud shadow
    3 - Snow
    4 - Cloud
    """
    fmask = ee.Image(img.select(['fmask']))
    cloud_mask = fmask.lt(2)
    # cloud_mask = fmask.eq(2).Or(fmask.eq(3)).Or(fmask.eq(4)).Not()
    return img \
        .select(refl_toa_bands + refl_sur_bands + ['lst']) \
        .updateMask(cloud_mask) \
        .addBands(img.select(['cloud_score', 'fmask'])) \
        .copyProperties(img, system_properties)


def pair_func(elev_image):
    """Elevation based air pressure"""
    return elev_image.expression(
        '101.3 * pow((293 - 0.0065 * b()) / 293, 5.26)')


# def prism_ppt_func(prism_image):
#     """PRISM water year precipitation
#
#     Depends on maps engine assets
#     """
#     return prism_image.select([0], ['ppt']) \
#         .copyProperties(prism_image, system_properties)


# DEADBEEF - Using GRIDMET precipitation band directly
# def gridmet_ppt_func(gridmet_image):
#     """GRIDMET daily precipitation"""
#     return gridmet_image.select(['pr'], ['ppt']).max(0) \
#         .copyProperties(gridmet_image, system_properties)


# DEADBEEF - Using GRIDMET ETo/ETr bands directly
# DEADBEEF - Not using geerefet for ETo/ETr calculation
# def gridmet_eto_func(gridmet_image):
#     """GRIDMET Daily ETo"""
#     return ee.Image(geerefet.Daily.gridmet(gridmet_image).eto()).max(0)\
#         .copyProperties(gridmet_image, system_properties)
#
# def gridmet_etr_func(gridmet_image):
#     """GRIDMET Daily ETr"""
#     return ee.Image(geerefet.Daily.gridmet(gridmet_image).etr()).max(0)\
#         .copyProperties(gridmet_image, system_properties)


# DEADBEEF - Using GRIDMET ETo/ETr bands directly
# DEADBEEF - Not computing ETo/ETr from component bands
# def gridmet_etr_func(gridmet_image):
#     """GRIDMET Daily ETr"""
#     scene_date = ee.Algorithms.Date(gridmet_image.get('system:time_start'))
#     doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
#
#     # Read in GRIDMET layers
#     tmin = gridmet_image.select(['tmmn']).subtract(273.15)  # K to C
#     tmax = gridmet_image.select(['tmmx']).subtract(273.15)  # K to C
#     # rhmin = gridmet_image.select(['rmin']).multiply(0.01)  # % to decimal
#     # rhmax = gridmet_image.select(['rmax']).multiply(0.01)  # % to decimal
#     q = gridmet_image.select(['sph'])                      # kg kg-1
#     rs = gridmet_image.select(['srad']).multiply(0.0864)   # W m-2 to MJ m-2 day-1
#     uz = gridmet_image.select(['vs'])                      # m/s?
#     zw = 10.0    # Windspeed measurement/estimated height (GRIDMET=10m)
#
#     # Vapor pressure from RHmax and RHmin (Eqn 11)
#     # ea = es_tmin.multiply(rhmax).add(es_tmax.multiply(rhmin)).multiply(0.5)
#     # Vapor pressure from specific humidity (Eqn )
#     # To match standardized form, ea is calculated from elevation based pair
#     pair = pair_func(ee.Image('USGS/NED'))
#     ea = pair.expression(
#         'q * pair / (0.622 + 0.378 * q)', {'pair': pair, 'q': q})
#
#     return daily_pet_func(
#         doy, tmin, tmax, ea, rs, uz, zw, 1600, 0.38).copyProperties(
#             gridmet_image, system_properties)
#
# def gridmet_eto_func(gridmet_image):
#     """GRIDMET Daily ETo"""
#     scene_date = ee.Algorithms.Date(gridmet_image.get('system:time_start'))
#     doy = ee.Number(scene_date.getRelative('day', 'year')).add(1).double()
#
#     # Read in GRIDMET layers
#     tmin = gridmet_image.select(['tmmn']).subtract(273.15)  # K to C
#     tmax = gridmet_image.select(['tmmx']).subtract(273.15)  # K to C
#     # rhmin = gridmet_image.select(['rmin']).multiply(0.01)  # % to decimal
#     # rhmax = gridmet_image.select(['rmax']).multiply(0.01)  # % to decimal
#     q = gridmet_image.select(['sph'])                      # kg kg-1
#     rs = gridmet_image.select(['srad']).multiply(0.0864)   # W m-2 to MJ m-2 day-1
#     uz = gridmet_image.select(['vs'])                      # m/s?
#     zw = 10.0  # Windspeed measurement/estimated height (GRIDMET=10m)
#
#     # Vapor pressure from RHmax and RHmin (Eqn 11)
#     # ea = es_tmin.multiply(rhmax).add(es_tmax.multiply(rhmin)).multiply(0.5)
#     # Vapor pressure from specific humidity (Eqn )
#     # To match standardized form, ea is calculated from elevation based pair
#     pair = pair_func(ee.Image('USGS/NED'))
#     ea = pair.expression(
#         'q * pair / (0.622 + 0.378 * q)', {'pair': pair, 'q': q})
#
#     return daily_pet_func(
#         doy, tmin, tmax, ea, rs, uz, zw, 900, 0.34).copyProperties(
#             gridmet_image, system_properties)
#
#
# def daily_pet_func(doy, tmin, tmax, ea, rs, uz, zw, cn=900, cd=0.34):
#     """Daily ASCE Penman Monteith Standardized Reference ET
#
#     Daily ETo cn=900, cd=0.34
#     Daily ETr cn=1600, cd=0.38
#
#     doy -- day of year
#     tmin -- minimum daily temperature [C]
#     tmax -- maximum daily temperature [C]
#     ea -- vapor pressure [?]
#     rs -- incoming solar radiation [MJ m-2 day]
#     uz -- wind speed [m s-1]
#     zw -- wind speed height [m]
#     cn -- coefficient
#     cd -- coefficient
#
#     """
#     # Globals in playground/javascript
#     pi = math.pi
#     lat = ee.Image.pixelLonLat().select(['latitude']).multiply(pi / 180)
#     lon = ee.Image.pixelLonLat().select(['longitude']).multiply(pi / 180)
#     elev = ee.Image('USGS/NED')
#     pair = pair_func(elev)
#
#     # Calculations
#     tmean = tmin.add(tmax).multiply(0.5)  # C
#     psy = pair.multiply(0.000665)
#     es_tmax = vapor_pressure_func(tmax)  # C
#     es_tmin = vapor_pressure_func(tmin)  # C
#     es_tmean = vapor_pressure_func(tmean)
#     es_slope = es_tmean.expression(
#         '4098 * es / (pow((t + 237.3), 2))', {'es': es_tmean, 't': tmean})
#     es = es_tmin.add(es_tmax).multiply(0.5)
#
#     # Extraterrestrial radiation (Eqn 24, 27, 23, 21)
#     delta = ee.Image.constant(
#         doy.multiply(2 * pi / 365).subtract(1.39435).sin().multiply(0.40928))
#     omegas = lat.expression(
#         'acos(-tan(lat) * tan(delta))', {'lat': lat, 'delta': delta})
#     theta = omegas.expression(
#         'omegas * sin(lat) * sin(delta) + cos(lat) * cos(delta) * sin(b())',
#         {'omegas': omegas, 'lat': lat, 'delta': delta})
#     dr = ee.Image.constant(
#         doy.multiply(2 * pi / 365).cos().multiply(0.033).add(1))
#     ra = theta.expression(
#         '(24 / pi) * gsc * dr * theta',
#         {'pi': pi, 'gsc': 4.92, 'dr': dr, 'theta': theta})
#
#     # Simplified clear sky solar formulation (Eqn 19)
#     # var rso = elev.expression(
#     #     '(0.75 + 2E-5 * elev) * ra', {'elev':elev, 'ra':ra})
#
#     # This is the full clear sky solar formulation
#     # sin of the angle of the sun above the horizon (D.5 and Eqn 62)
#     sin_beta_24 = lat.expression(
#         'sin(0.85 + 0.3 * lat * delta / 0.40928 - 0.42 * lat ** 2)',
#         {'lat': lat, 'delta': delta})
#
#     # Precipitable water (Eqn D.3)
#     w = pair.expression(
#         '0.14 * ea * pair + 2.1', {'pair': pair, 'ea': ea})
#
#     # Clearness index for direct beam radiation (Eqn D.2)
#     # Limit sin_beta >= 0.01 so that KB does not go undefined
#     kb = pair.expression(
#         '0.98 * exp((-0.00146 * pair) / (kt * sin_beta) - '
#         '0.075 * pow((w / sin_beta), 0.4))',
#         {'pair': pair, 'kt': 1.0, 'sin_beta': sin_beta_24.max(0.01), 'w': w})
#
#     # Transmissivity index for diffuse radiation (Eqn D.4)
#     kd = kb.multiply(-0.36).add(0.35).min(kb.multiply(0.82).add(0.18))
#     # var kd = kb.multiply(-0.36).add(0.35)
#     #     .where(kb.lt(0.15), kb.multiply(0.82).add(0.18))
#
#     # (Eqn D.1)
#     rso = ra.multiply(kb.add(kd))
#     # Cloudiness fraction (Eqn 18)
#     fcd = rs.divide(rso).clamp(0.3, 1).multiply(1.35).subtract(0.35)
#
#     # Net long-wave radiation (Eqn 17)
#     rnl = ea.expression(
#         ('4.901E-9 * fcd * (0.34 - 0.14 * sqrt(ea)) * '
#          '(pow(tmax_k, 4) + pow(tmin_k, 4)) / 2'),
#         {'ea': ea, 'fcd': fcd,
#          'tmax_k': tmax.add(273.15), 'tmin_k': tmin.add(273.15)})
#
#     # Net radiation (Eqns 15 and 16)
#     rn = rs.multiply(0.77).subtract(rnl)
#
#     # Wind speed (Eqn 33)
#     u2 = uz.expression('b() * 4.87 / log(67.8 * zw - 5.42)', {'zw': zw})
#
#     # Daily ETo (Eqn 1)
#     return tmin.expression(
#         '(0.408 * slope * (rn - g) + (psy * cn * u2 * (es - ea) / (t + 273))) / '
#         '(slope + psy * (cd * u2 + 1))',
#         {'slope': es_slope, 'rn': rn, 'g': 0, 'psy': psy, 'cn': cn,
#          't': tmean, 'u2': u2, 'es': es, 'ea': ea, 'cd': cd})


# def vapor_pressure_func(temperature_image):
#     """Vapor Pressure
#
#     in kPa with temperature in C
#     """
#     return temperature_image.expression(
#         '0.6108 * exp(17.27 * b() / (b() + 237.3))')


def mosaic_landsat_images(landsat_coll, mosaic_method):
    """"""
    def mosaic_id_func(image):
        """Set MOSAIC_ID with row set to XXX

        Using GEE Collection 1 system:index naming convention
        LT05_PPPRRR_YYYYMMDD
        """
        scene_id = ee.String(ee.Image(image).get('system:index'))
        mosaic_id = scene_id.slice(0, 8).cat('XXX').cat(scene_id.slice(11, 20))
        # If mosaicing after merging, SCENE_ID is at end
        # scene_id = ee.String(ee.List(ee.String(
        #     image.get('system:index')).split('_')).get(-1))
        # Build product ID from old style scene ID
        # scene_id = ee.String(img.get('system:index'))
        # scene_id = scene_id.slice(0, 2).cat('0') \
        #     .cat(scene_id.slice(2, 3)).cat('_') \
        #     .cat(scene_id.slice(3, 9)).cat('_') \
        #     .cat(ee.Date(img.get('system:time_start')).format('yyyyMMdd'))
        return image.setMulti({'MOSAIC_ID': mosaic_id})
    landsat_mosaic_coll = ee.ImageCollection(
        landsat_coll.map(mosaic_id_func))

    mosaic_id_list = ee.List(ee.Dictionary(ee.FeatureCollection(
        landsat_mosaic_coll.aggregate_histogram('MOSAIC_ID'))).keys())

    def set_mosaic_id(mosaic_id):
        return ee.Feature(None, {'MOSAIC_ID': ee.String(mosaic_id)})
    mosaic_id_coll = ee.FeatureCollection(mosaic_id_list.map(set_mosaic_id))

    join_coll = ee.Join.saveAll('join').apply(
        mosaic_id_coll, landsat_mosaic_coll,
        ee.Filter.equals(leftField='MOSAIC_ID', rightField='MOSAIC_ID'))

    def aggregate_func(ftr):
        # The composite image time will be 0 UTC (not Landsat time)
        coll = ee.ImageCollection.fromImages(ftr.get('join'))
        time = ee.Image(ee.List(ftr.get('join')).get(0)).get('system:time_start')
        if mosaic_method == 'mean':
            image = coll.mean()
        elif mosaic_method == 'median':
            image = coll.median()
        elif mosaic_method == 'mosaic':
            image = coll.mosaic()
        elif mosaic_method == 'min':
            image = coll.min()
        elif mosaic_method == 'max':
            image = coll.max()
        else:
            image = coll.first()
        return ee.Image(image).setMulti({
            'SCENE_ID': ee.String(ftr.get('MOSAIC_ID')),
            'system:time_start': time})
    mosaic_coll = ee.ImageCollection(join_coll.map(aggregate_func))

    return mosaic_coll
