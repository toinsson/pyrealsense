# -*- coding: utf-8 -*-
# Licensed under the Apache-2.0 License, see LICENSE for details.

"""This module loads rsutilwrapper and librealsense library."""

import ctypes
import sys
import os
import warnings

os_name = sys.platform
lrs_prefix_mapping = {'darwin': 'lib', 'linux': 'lib', 'linux2': 'lib', 'win32': ''}
lrs_suffix_mapping = {'darwin': '.dylib', 'linux': '.so', 'linux2': '.so', 'win32': '.dll'}

try:
    lrs_prefix = lrs_prefix_mapping[os_name]
    lrs_suffix = lrs_suffix_mapping[os_name]
except KeyError:
    raise OSError('OS not supported.')


## import C lib
try:
    lrs = ctypes.CDLL(lrs_prefix+'realsense'+lrs_suffix)
except OSError:
    warnings.warn("librealsense not found.")
    lrs = None

## try import since docs will crash here
try:
    from . import rsutilwrapper
except ImportError:
    warnings.warn("rsutilwrapper not found.")
    rsutilwrapper = None

