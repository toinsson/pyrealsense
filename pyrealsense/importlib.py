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

    for f in os.listdir(dirname):
        if f.endswith(rsu_suffix):
            f_name = f

    return os.path.join(dirname, f_name)

rsutilwrapper = ctypes.CDLL(_find_extension_name())

## import C lib
lrs = ctypes.CDLL(lrs_prefix+'realsense'+lrs_suffix)

