# import glob
# import os
# import logging

import pytest

import ee_tools.utils as utils


# def test_get_ini_path(workspace):
#     import Tkinter, tkFileDialog
#     assert False


@pytest.mark.parametrize(
    "inputstr,expected",
    [
        ["2-4, 7, 9-10", set([2, 3, 4, 7, 9, 10])],
        ["", set()],
        ["A", set()]
    ]
)
def test_parse_int_set(inputstr, expected):
    """Return list of numbers given a string of ranges

    Originally in utils.py

    http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html
    """
    assert utils.parse_int_set(inputstr) == expected


# def test_remove_file(file_path):
#     """Remove a feature/raster and all of its anciallary files"""
#     assert False


# def test_valid_file(parser, arg):
#     assert False


@pytest.mark.parametrize(
    "start,end,expected",
    [
        [1, 12, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        [10, 9, [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        [3, 5, [3, 4, 5]],
        [10, 1, [10, 11, 12, 1]]
    ]
)
def test_month_range(start, end, expected):
    assert list(utils.month_range(start, end)) == expected


@pytest.mark.parametrize(
    "a,b,x_min,x_max,expected",
    [
        [1, 12, 1, 12, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        [None, None, 1, 12, []],
        [None, 12, 1, 12, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        [1, None, 1, 12, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]],
        [10, 9, 1, 12, [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        [3, 5, 1, 12, [3, 4, 5]],
        [10, 1, 1, 12, [10, 11, 12, 1]]
    ]
)
def test_wrapped_range(a, b, x_min, x_max, expected):
    """Return the values between a range b for a given start/end"""
    assert list(utils.wrapped_range(a, b, x_min, x_max)) == expected
