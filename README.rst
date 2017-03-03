PyRealsense
===========

Simple `ctypes <https://docs.python.org/2/library/ctypes.html>`__
extension to the
`librealsense <https://github.com/IntelRealSense/librealsense>`__
library for Linux and Mac OS.

Dependencies
------------

The library depends on
`pycparser <https://github.com/eliben/pycparser>`__ for parsing the
librealsense h files and extracting necessary enums and structures
definitions. `Numpy <http://www.numpy.org/>`__ is used for generic data
shuffling.

Installation
------------

::

    pip install pycparser numpy
    python setup.py install

Online Usage
------------

::

    ## setup logging
    import logging
    logging.basicConfig(level = logging.INFO)

    ## import the package
    import pyrealsense as pyrs

    ## start the service
    pyrs.start()

    ## create a device from device id and streams of interest
    cam = pyrs.Device(device_id = 0, streams = [pyrs.ColourStream(fps = 60)])

    ## wait for data and retrieve numpy array for ~1 second
    for i in range(60):
        cam.wait_for_frame()
        print(cam.colour)

The server for Realsense devices is started with ``pyrs.start()`` which
will printout the number of devices available.

Different devices can be created from the ``Device`` factory. They are
created as their own class defined by device id, name, serial, firmware,
as well as streams passed and camera presets. The default behaviour
create a device with ``id = 0`` and setup the colour, depth, pointcloud
and colour\_aligned\_depth streams.

The available streams are either native or synthetic, and each one will
create a property that exposes the current content of the frame buffer
in the form of ``device.<stream_name>``, where ``<stream_name>`` is
colour, depth, points, cad or dac. To get access to new data,
``Device.wait_for_frame`` has to be called once per frame.

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

The module ``offline`` allows storing the rs\_intrinsics and
depth\_scale of a device to disk, by default in the home directory in
the file ``.pyrealsense``. This can later be loaded and used to
deproject depth data into pointcloud, which is useful to store raw video
file and save some disk memory.

Examples
--------

The examples are split based on the visualisation technology they
require. One shows a still image with
`matplotlib <http://matplotlib.org/>`__, another one streams depth and
color data with `opencv <http://opencv.org/>`__, and the last one
displays a live feed of the pointcloud with
`VTK <http://www.vtk.org/>`__.

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

-  support for Windows
-  support for several cameras in offline module

.. |Build Status| image:: https://travis-ci.org/toinsson/pyrealsense.svg?branch=master
   :target: https://travis-ci.org/toinsson/pyrealsense
