# Pyrealsense

Simple [ctypes](https://docs.python.org/2/library/ctypes.html) extension to the [librealsense](https://github.com/IntelRealSense/librealsense) library.

## Dependencies

The library depends on [pycparser](https://github.com/eliben/pycparser) for parsing the librealsense h files and extracting necessary enums and structures definitions. Numpy is used for generic data shuffling.

## Installation

    python setup.py install

## Usage

    ## setup logging
    import logging
    logging.basicConfig(level=logging.INFO)

    ## import the package
    import pyrealsense as pyrs

    ## start the service
    pyrs.start()

    ## create a device from device id and streams of interest
    cam = pyrs.Device(device_id = 0, streams = [pyrs.ColourStream(fps=60)])

    ## wait for data and retrieve numpy array
    cam.wait_for_frame()
    print(cam.colour)

The server for Realsense devices is started with `pyrs.start()` which will printout the number of devices available.

Different devices can be created from the `Device` class. They are defined by device id and streams passed on creation. The default behaviour create a device with `id = 0` and setup the colour, depth, pointcloud and colour_aligned_depth streams.

The available streams are either native or synthetic, and each one will create a property on the Device instance that exposes the current content of the frame buffer in the form of `device.<stream_name>`, where `<stream_name>` is colour, depth, points, cad or dac. To get access to new data, `Device.wait_for_frame` has to be called.

## Caveats
To this point, this wrapper has only been tested with:
- Ubuntu 16.04 LTS
- Python 2.7
- SR300 camera
- [librealsense v1.9.7](https://github.com/IntelRealSense/librealsense/tree/v1.9.7)

## Build Status
Ubuntu Trusty, python 2 and 3: [![Build Status](https://travis-ci.org/toinsson/pyrealsense.svg?branch=master)](https://travis-ci.org/toinsson/pyrealsense)
