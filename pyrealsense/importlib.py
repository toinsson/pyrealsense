# -*- coding: utf-8 -*-
# Licensed under the Apache-2.0 License, see LICENSE for details.

"""This module loads rsutilwrapper and librealsense library."""

import ctypes
import sys
import os

os_name = sys.platform
lrs_prefix_mapping = {'darwin': 'lib', 'linux': 'lib', 'linux2': 'lib', 'win32': ''}
lrs_suffix_mapping = {'darwin': '.dylib', 'linux': '.so', 'linux2': '.so', 'win32': '.dll'}
rsu_suffix_mapping = {'darwin': '.so', 'linux': '.so', 'linux2': '.so', 'win32': '.pyd'}

try:
    lrs_prefix = lrs_prefix_mapping[os_name]
    lrs_suffix = lrs_suffix_mapping[os_name]
    rsu_suffix = rsu_suffix_mapping[os_name]
except KeyError:
    raise OSError('OS not supported.')

## hacky way to load "extension" module
def _find_extension_name():
    dirname = os.path.dirname(__file__)
    f_name = ''
    for f in os.listdir(dirname):
        if f.endswith(rsu_suffix):
            f_name = f
    return os.path.join(dirname, f_name)

## prevent crash for Sphinx when extension is not compiled before hand
try:
    rsutilwrapper = ctypes.CDLL(_find_extension_name())
except OSError:
    import warnings
    warnings.warn("rsutilwrapper not found.")
    rsutilwrapper = None

## import C lib
try:
    lrs = ctypes.CDLL(lrs_prefix+'realsense'+lrs_suffix)
except OSError:
    import warnings
    warnings.warn("librealsense not found.")
    lrs = None

