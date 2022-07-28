import argparse
import datetime
import glob
import logging
import os
import re
import subprocess
import sys
from time import sleep
# Python 3 (or 2 with future module)
import tkinter
import tkinter.filedialog
# Python 2
# import Tkinter
# import tkFileDialog

import ee

import ee_tools.wrs2 as wrs2

ee.Initialize()


# def arg_valid_date(date_str):
#     """Argparse specific function for validating date strings"""
#     try:
#         datetime.datetime.strptime(date_str, '%Y-%m-%d')
#         return date_str
#     except ValueError:
#         raise argparse.ArgumentTypeError(
#             '{} is an invalid date'.format(date_str))


def arg_valid_file(file_path):
    """Argparse specific function for testing if file exists

    Convert relative paths to absolute paths
    """
    if os.path.isfile(os.path.abspath(os.path.realpath(file_path))):
        return os.path.abspath(os.path.realpath(file_path))
        # return file_path
    else:
        raise argparse.ArgumentTypeError('{} does not exist'.format(file_path))


def date_range(start_date, end_date):
    """Yield datetimes within a date range

    Args:
        start_date (str): ISO format start date (YYYY-MM-DD)
        end_date (str): ISO format end date (YYYY-MM-DD).
            End date will NOT be included in range (exclusive).

    Yields
        datetime:

    """
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    for n in range(int((end_dt + datetime.timedelta(1) - start_dt).days)):
        yield (start_dt + datetime.timedelta(n)).strftime('%Y-%m-%d')


# def date_range(start_dt, end_dt, days=1, skip_leap_days=True):
#     """Generate dates within a range (inclusive)
#
#     Args:
#         start_dt (datetime): start date
#         end_dt (datetime): end date
#         days (int, optional): step size. Defaults to 1.
#         skip_leap_days (bool, optional): if True, skip leap days while incrementing.
#             Defaults to True.
#
#     Yields
#         datetime:
#
#     """
#     import copy
#     curr_dt = copy.copy(start_dt)
#     while curr_dt <= end_dt:
#         if not skip_leap_days or curr_dt.month != 2 or curr_dt.day != 29:
#             yield curr_dt
#         curr_dt += datetime.timedelta(days=days)


def get_buckets(project_name):
    """Return Google Cloud Storage buckets associated with project

    Args:
        project_name (str): AppEngine project name

    Returns:
        list of bucket names

    """
    # logging.debug('\nGetting cloud storage bucket list')
    try:
        bucket_list = subprocess.check_output(
            ['gsutil', 'ls', '-p', project_name],
            universal_newlines=True, shell=True)
    except Exception as e:
        logging.error(
            '\nERROR: There was a problem getting the bucket list ' +
            'using gsutil, exiting')
        logging.error('  Exception: {}'.format(e))
        sys.exit()
    bucket_list = [
        b.replace('gs://', '').replace('/', '')
        for b in filter(None, re.split(r'[~\r\n]+', bucket_list))]
    logging.info('  {:16s} {}'.format('Buckets:', ', '.join(bucket_list)))
    return bucket_list


def get_bucket_files(project_name, bucket_name):
    """Return Google Cloud Storage buckets associated with project

    Args:
        project_name (str): AppEngine project name
        bucket_name (str): Google Storage bucket name

    Returns:
        list of file names

    """
    try:
        file_list = subprocess.check_output(
            ['gsutil', 'ls', '-r', '-p', project_name, bucket_name],
            universal_newlines=True, shell=True)
    except Exception as e:
        logging.error(
            '\nERROR: There was a problem getting the bucket file list ' +
            'using gsutil, exiting')
        logging.error('  Exception: {}'.format(e))
        sys.exit()
    return file_list


def get_ee_tasks(states=['RUNNING', 'READY']):
    """Return current active tasks

    Returns:
        dict of task descriptions (key) and task IDs (value)

    """

    logging.debug('\nActive Tasks')
    tasks = {}
    task_list = sorted([
        [t['state'], t['description'], t['id']]
        for t in ee.data.getTaskList()
        if t['state'] in states])
    if task_list:
        logging.debug('  {:8s} {}'.format('STATE', 'DESCRIPTION'))
        logging.debug('  {:8s} {}'.format('=====', '==========='))
    else:
        logging.debug('  None')
    for t_state, t_desc, t_id in task_list:
        logging.debug('  {:8s} {}'.format(t_state, t_desc))
        tasks[t_desc] = t_id
        # tasks[t_id] = t_desc
    return tasks


def get_ini_path(workspace):
    """Open dialog box to allow user to select an .ini file"""
    # Python 3 (or 2 with future module)
    root = tkinter.Tk()
    ini_path = tkinter.filedialog.askopenfilename(
        initialdir=workspace, parent=root, filetypes=[('INI files', '.ini')],
        title='Select the target INI file')
    # Python 2
    # root = Tkinter.Tk()
    # ini_path = tkFileDialog.askopenfilename(
    #     initialdir=workspace, parent=root, filetypes=[('INI files', '.ini')],
    #     title='Select the target INI file')
    root.destroy()
    return ini_path


def ee_getinfo(ee_obj, n=10):
    """Make an exponential backoff getInfo call on the Earth Engine object"""
    output = None
    for i in range(1, n):
        try:
            output = ee_obj.getInfo()
        except Exception as e:
            logging.info('    Resending query ({}/10)'.format(i))
            logging.debug('    {}'.format(e))
            sleep(i ** 2)
        if output:
            break
    return output


def ee_request(request_obj, n=10):
    """Make an exponential backoff Earth Engine request"""
    output = None
    for i in range(1, n):
        try:
            output = request_obj
        except Exception as e:
            logging.info('    Resending query ({}/10)'.format(i))
            logging.debug('    {}'.format(e))
            sleep(i ** 2)
        if output:
            break
    return output


def parse_int_set(nputstr=""):
    """Return list of numbers given a string of ranges

    http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html
    """
    selection = set()
    invalid = set()

    # Tokens are comma seperated values
    # AttributeError will get raised when nputstr is empty
    try:
        tokens = [x.strip() for x in nputstr.split(',')]
    except AttributeError:
        return set()

    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token) - 1]
                    for x in range(first, last + 1):
                        selection.add(x)
            except:
                # not an int and not a range...
                invalid.add(i)
    # Report invalid tokens before returning valid selection
    # print "Invalid set: " + str(invalid)
    return selection


def wrs2_tile_geom_func(tile_list):
    """"""
    geom_list = [
        wrs2.tile_centroids[tile]
        for tile in tile_list
        if tile in wrs2.tile_centroids.keys()]
    return ee.Geometry.MultiPoint(geom_list, 'EPSG:4326')


def remove_file(file_path):
    """Remove a feature/raster and all of its anciallary files"""
    file_ws = os.path.dirname(file_path)
    for file_name in glob.glob(os.path.splitext(file_path)[0] + ".*"):
        os.remove(os.path.join(file_ws, file_name))


def month_range(start, end):
    """Generate month numbers between start and end, wrapping if necessary

    Equivalent to wrapped_range(start, end, x_min=1, x_max=12)

    Args:
        start (int): Start month
        end (int): End month

    Yields:
        int: The next month number

    Examples:
        >>> month_range(1, 12))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> month_range(10, 9))
        [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> month_range(3, 5))
        [3, 4, 5]
        >>> month_range(10, 1))
        [10, 11, 12, 1]

    """
    m = int(start)
    while True:
        yield m
        if m == end:
            break
        m += 1
        if m > 12:
            m = 1


def wrapped_range(start, end, x_min=1, x_max=12):
    """Return the values between a range b for a given start/end

    Args:
        start (int): Start value
        end (int): End value
        x_min (int): Minimum value
        x_max (int): Maximum value

    Yields:
        int: The next number in the wrapped range

    Examples:
        >>> wrapped_range(1, 12, 1, 12))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> wrapped_range(None, None, 1, 12))
        []
        >>> wrapped_range(None, 12, 1, 12))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> wrapped_range(1, None, 1, 12))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        >>> wrapped_range(10, 9, 1, 12))
        [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> wrapped_range(3, 5, 1, 12))
        [3, 4, 5]
        >>> wrapped_range(10, 1, 1, 12))
        [10, 11, 12, 1]

    """
    if start is None and end is None:
        return
    if start is None:
        start = x_min
    if end is None:
        end = x_max

    x = int(start)
    while True:
        yield x
        if x == end:
            break
        x += 1
        if x > x_max:
            x = x_min


def unique_keep_order(seq):
    """https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates
       -from-a-list-in-whilst-preserving-order?page=1&tab=active#tab-top
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
