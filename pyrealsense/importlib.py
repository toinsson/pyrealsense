import ctypes
import sys
import os

os_name = sys.platform
lrs_suffix_mapping = {'darwin':'.dylib', 'linux':'.so', 'win':'.dll'}
rsu_suffix_mapping = {'darwin':'.so', 'linux':'.so', 'win':'.dll'}

try:
    lrs_suffix = lrs_suffix_mapping[os_name]
    rsu_suffix = rsu_suffix_mapping[os_name]
except KeyError:
    print('OS not supported.')

print ('rsu_suffix: ', rsu_suffix)

## hacky way to load "extension" module
_DIRNAME = os.path.dirname(__file__)

print('dir:', _DIRNAME)
for _file in os.listdir(_DIRNAME):
    print('l:, ', _file, _file.__class__)
    if _file.endswith(rsu_suffix):
        _rsutilwrapper = _file

_tmp = os.path.join(_DIRNAME, _rsutilwrapper)

rsutilwrapper = ctypes.CDLL(_tmp)

## import C lib
lrs = ctypes.CDLL('librealsense'+lrs_suffix)

