Readme
======

Cross-platform
`ctypes <https://docs.python.org/2/library/ctypes.html>`__/`Cython <http://cython.org/>`__
wrapper to the
`librealsense <https://github.com/IntelRealSense/librealsense>`__
library.

Prerequisites
-------------

-  install
   `librealsense <https://github.com/IntelRealSense/librealsense#installation-guide>`__
   and run the examples.

-  install the dependencies: pyrealsense uses
   `pycparser <https://github.com/eliben/pycparser>`__ for extracting
   necessary enums and structures definitions from the librealsense API,
   `Cython <http://cython.org/>`__ for wrapping the inlined functions in
   the librealsense API, and `Numpy <http://www.numpy.org/>`__ for
   generic data shuffling.

-  Windows specifics: set environment variable PYRS\_INCLUDES to the
   ``rs.h`` directory location and environment variable PYRS\_LIBS to
   the librealsense binary location. You might also need to have
   ``stdint.h`` available in your path.

Installation
------------

from `PyPI <https://pypi.python.org/pypi/pyrealsense/2.0>`__ - (OBS: not
always the latest):

::

    pip install pyrealsense

from source:

::

    python setup.py install

Online Usage
------------

::

    ## setup logging
    import logging
    logging.basicConfig(level = logging.INFO)

    ## import the package
    import pyrealsense as pyrs

    ## start the service - also available as context manager
    serv = pyrs.Service()

    ## create a device from device id and streams of interest
    cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = 60)])

    ## retrieve 60 frames of data
    for _ in range(60):
        cam.wait_for_frames()
        print(cam.color)

    ## stop camera and service
    cam.stop()
    serv.stop()

The server for Realsense devices is started with ``pyrs.Service()``
which will printout the number of devices available. It can also be
started as a context with ``with pyrs.Service():``.

Different devices can be created from the service ``Device`` factory.
They are created as their own class defined by device id, name, serial,
firmware as well as enabled streams and camera presets. The default
behaviour create a device with ``id = 0`` and setup the color, depth,
pointcloud, color\_aligned\_depth, depth\_aligned\_color and infrared
streams.

The available streams are either native or synthetic, and each one will
create a property that exposes the current content of the frame buffer
in the form of ``device.<stream_name>``, where ``<stream_name>`` is
color, depth, points, cad, dac or infrared. To get access to new data,
``Device.wait_for_frames`` has to be called once per frame.

Offline Usage
-------------

::

    ## with connected device cam
    from pyrealsense import offline
    offline.save_depth_intrinsics(cam)

::

    ## previous device cam now offline
    from pyrealsense import offline
    offline.load_depth_intrinsics('610205001689')  # camera serial number
    d = np.linspace(0, 1000, 480*640, dtype=np.uint16)
    pc = offline.deproject_depth(d)

The module ``offline`` can store the rs\_intrinsics and depth\_scale of
a device to disk by default in the user's home directory in the file
``.pyrealsense``. This can later be loaded and used to deproject depth
data into pointcloud, which is useful to store raw video file and save
some disk memory.

Examples
--------

There are 3 examples using different visualisation technologies: - still
color with `matplotlib <http://matplotlib.org/>`__ - color and depth
stream with `opencv <http://opencv.org/>`__ - pointcloud stream with
`VTK <http://www.vtk.org/>`__

Caveats
-------

To this point, this wrapper is tested with:

-  `librealsense
   v1.12.1 <https://github.com/IntelRealSense/librealsense/tree/v1.12.1>`__
-  Ubuntu 16.04 LTS, Mac OS X 10.12.2 w/ SR300 camera
-  Mac OS X 10.12.3 w/ R200 camera

The offline module only supports a single camera.

Build Status
------------

Ubuntu Trusty, python 2 and 3: |Build Status|

Possible Pull Requests
----------------------

-  improvments to the documentation
-  more functionality from ``rs.h``
-  boiler plate code (Qt example?)
-  support for several cameras in offline module
-  continuous integration for Windows and MacOs

Make sure to push to the ``dev`` branch.

.. |Build Status| image:: https://travis-ci.org/toinsson/pyrealsense.svg?branch=master
   :target: https://travis-ci.org/toinsson/pyrealsense
