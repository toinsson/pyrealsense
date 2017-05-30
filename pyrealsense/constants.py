import pycparser
import io

# Platform dependent
import os

import sys
from os import environ, path

## construct path to librealsense/rs.h
## for docs, use locally stored
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    rs_h_filename = './rs.h'
## for windows, use PYRS_INCLUDE
elif sys.platform == 'win32':

    if 'PYRS_INCLUDES' not in environ:
        raise Exception('PYRS_INCLUDES must be set to the location of the librealsense headers!')
    else:
        rs_h_filename = path.join(environ['PYRS_INCLUDES'], 'rs.h')
## on other platforms, fall back on default location
else:
    rs_h_filename = '/usr/local/include/librealsense/rs.h'

## if not found, exit
if not path.exists(rs_h_filename):
    raise Exception('librealsense/rs.h header not found at {}'.format(rs_h_filename))


# Dynamically extract API version
api_version = 0
with io.open(rs_h_filename, encoding='latin') as rs_h_file:
    for l in rs_h_file.readlines():
        if 'RS_API' in l:
            key, val = l.split()[1:]
            globals()[key] = val
            api_version = api_version * 100 + int(val)
        if api_version >= 10000:
            break
    globals()['RS_API_VERSION'] = api_version


def _get_enumlist(obj):
    for _, cc in obj.children():
        if type(cc) is pycparser.c_ast.EnumeratorList:
            return cc
        else:
            return _get_enumlist(cc)


# Dynamically generate classes
ast = pycparser.parse_file(rs_h_filename, use_cpp=True)
for c in ast.ext:
    if c.name in ['rs_capabilities',
                  'rs_stream',
                  'rs_format',
                  'rs_distortion',
                  'rs_ivcam_preset',
                  'rs_option']:
        e = _get_enumlist(c)

        class_name = c.name
        class_dict = {}
        for i, (_, child) in enumerate(e.children()):
            class_dict[child.name] = i

        # Generate the class and add to global scope
        class_gen = type(class_name, (object,), class_dict)
        globals()[class_name] = class_gen
        del class_gen

