from setuptools import setup, Extension
from setuptools import find_packages
from os import path
import io

import numpy as np

here = path.abspath(path.dirname(__file__))
with io.open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

module = [Extension('pyrealsense.rsutilwrapper',
                    sources=['pyrealsense/rsutilwrapper.c'],
                    libraries=['realsense'],
                    include_dirs=[np.get_include(), '/usr/local/include/librealsense'],
                    library_dirs=['/usr/local/lib'], )]

setup(name='pyrealsense',
      version='1.4',

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

