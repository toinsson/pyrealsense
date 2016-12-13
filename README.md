# Pyrealsense
Simple ctypes extension to the [librealsense](https://github.com/IntelRealSense/librealsense) library. 

## Dependencies

The library depends on [pycparser](https://github.com/eliben/pycparser) for parsing the librealsense h files and extract enums and structures definitions. Numpy is used for generic data shuffling.

## Installation

    python setup.py install

## Usage

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

The available streams are either native or synthetic, and each one will create a property on the Device instancve that exposes the current content of the frame buffer in the form of `device.<stream_name>`, where `<stream_name>` is colour, depth, points, cad or dac. To get access to new data, `Device.wait_for_frame` has to be called.

## caveats
To this point, this wrapper has only been tested with:
- Python 2.x
- Linux architecture
- SR300 camera

## build status
Ubuntu Trusty, python 2 and 3: [![Build Status](https://travis-ci.org/toinsson/pyrealsense.svg?branch=master)](https://travis-ci.org/toinsson/pyrealsense)
