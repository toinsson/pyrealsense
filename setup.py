from setuptools import find_packages
from os import path, environ
import io
import os
import re

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


# pip's single-source version method as described here:
# https://python-packaging-user-guide.readthedocs.io/single_source_version/
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# fetch include and library directories
inc_dirs = [np.get_include(), '/usr/local/include/librealsense']
lib_dirs = ['/usr/local/lib']

# windows environment variables
if 'PYRS_INCLUDES' in environ:
    inc_dirs.append(environ['PYRS_INCLUDES'])
if 'PYRS_LIBS' in environ:
    lib_dirs.append(environ['PYRS_LIBS'])

# cython extension, dont build if docs
on_rtd = environ.get('READTHEDOCS') == 'True'
if on_rtd:
    module = []
else:
    module = cythonize(
        [Extension(
            name='pyrealsense.rsutilwrapper',
            sources=["pyrealsense/rsutilwrapper.pyx", "pyrealsense/rsutilwrapperc.cpp"],
            libraries=['realsense'],
            include_dirs=inc_dirs,
            library_dirs=lib_dirs,
            language="c++",)])

# create long description from readme for pypi
here = path.abspath(path.dirname(__file__))
with io.open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='pyrealsense',
      version=find_version('pyrealsense', '__init__.py'),

      description='Cross-platform ctypes/Cython wrapper to the librealsense library.',
      long_description=long_description,
      author='Antoine Loriette',
      author_email='antoine.loriette@gmail.com',
      url='https://github.com/toinsson/pyrealsense',
      license='Apache',

      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        # 'License :: OSem :: Hardware',
      ],
      keywords='realsense',

      packages=find_packages(),
      ext_modules=module,
      setup_requires=['numpy', 'cython'],
      install_requires=['numpy', 'cython', 'pycparser', 'six'])
