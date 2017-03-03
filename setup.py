from setuptools import setup, Extension
from setuptools import find_packages
from os import path

import pycparser
import numpy as np
import io


def get_enumlist(obj):
    for cn, c in obj.children():
        if type(c) is pycparser.c_ast.EnumeratorList:
            return c
        else:
            return get_enumlist(c)


def write_enumlist(f, obj, name):
    classname = "class " + name + "(object):"
    enumerates = []
    for i, (cn, c) in enumerate(obj.children()):
        enumerates.append("    "+c.name + " = " + str(i))
    f.write("\n".join([classname] + enumerates) + "\n\n")


# do platform dependent
rs_h_filename = "/usr/local/include/librealsense/rs.h"

# extract RS_API in #define
rs_api = []
with io.open(rs_h_filename, encoding='latin') as rs_h_file:
    for l in rs_h_file.readlines():
        if "RS_API" in l:
            rs_api.append(" = ".join(l.split()[1:]))
        if len(rs_api) == 3:
            break
    versions = [int(x.split()[2]) for x in rs_api]
    rs_api.append("RS_API_VERSION = " + str((np.array(versions) * [10000, 100, 1]).sum()))
    with open("./pyrealsense/constants.py", "w") as constants:
        constants.write("\n".join(rs_api) + "\n\n")

ast = pycparser.parse_file(rs_h_filename, use_cpp=True)
with open("./pyrealsense/constants.py", "a") as constants:
    for c in ast.ext:
        if c.name in ['rs_capabilities',
                      'rs_stream',
                      'rs_format',
                      'rs_distortion',
                      'rs_ivcam_preset',
                      'rs_option']:
            e = get_enumlist(c)
            write_enumlist(constants, e, c.name)

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

module = [Extension('pyrealsense.rsutilwrapper',
                    sources=['pyrealsense/rsutilwrapper.c'],
                    libraries=['realsense'],
                    include_dirs=[np.get_include(), '/usr/local/include/librealsense'],
                    library_dirs=['/usr/local/lib'], )]

setup(name='pyrealsense',
      version='1.3',

      description='Simple ctypes extension to the librealsense library for Linux and Mac OS',
      long_description=long_description,
      url='https://github.com/toinsson/pyrealsense',
      license='Apache',

      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: System :: Hardware',
      ],
      keywords='realsense',

      packages=find_packages(),
      ext_modules=module,
      install_requires=['numpy', 'pycparser'],)