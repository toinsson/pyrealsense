from setuptools import setup, Extension
from setuptools import find_packages
from os import path, environ
import io
import sys


import numpy as np


here = path.abspath(path.dirname(__file__))
with io.open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


## fetch include and library directories
inc_dirs = [np.get_include(), '/usr/local/include/librealsense']
lib_dirs = ['/usr/local/lib']

if 'PYRS_INCLUDES' in environ:
    inc_dirs.append(environ['PYRS_INCLUDES'])
if 'PYRS_LIBS' in environ:
    lib_dirs.append(environ['PYRS_LIBS'])


## dont build extension if on RTD
import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    module = []
else:
    module = [Extension('pyrealsense.rsutilwrapper',
                    sources=['pyrealsense/rsutilwrapper.cpp'],
                    libraries=['realsense'],
                    include_dirs=inc_dirs,
                    library_dirs=lib_dirs, )]


setup(name='pyrealsense',
      version='1.5',

      description='Simple ctypes extension to the librealsense library for Linux and Mac OS',
      long_description=long_description,
      author='Antoine Loriette',
      author_email='antoine.loriette@gmail.com',
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
      setup_requires=['numpy',],
      install_requires=['numpy', 'pycparser'],)

