# PyRealsense

Cross-platform [ctypes](https://docs.python.org/2/library/ctypes.html)/[Cython](http://cython.org/) wrapper to the [librealsense](https://github.com/IntelRealSense/librealsense) C-library version 1.x. This wrapper is useful for legacy models such as SR300, F200 and R200.

OBS: there is no plan to support librealsense 2.x, as Intel already provides the Python binding through [pyrealsense2](https://pypi.org/project/pyrealsense2/).


## Prerequisites

- install [librealsense](https://github.com/IntelRealSense/librealsense#installation-guide) and run the examples.

- install the dependencies: pyrealsense uses [pycparser](https://github.com/eliben/pycparser) for extracting necessary enums and structures definitions from the librealsense API, [Cython](http://cython.org/) for wrapping the inlined functions in the librealsense API, and [Numpy](http://www.numpy.org/) for generic data shuffling.

- Windows specifics: set the environment variable PYRS_INCLUDES to the `rs.h` directory location and the environment variable PYRS_LIBS to the librealsense binary location. You might also need to have `stdint.h` available in your path.


## Installation

from [PyPI](https://pypi.python.org/pypi/pyrealsense/2.2) - (OBS: not always the latest):

    pip install pyrealsense

from source:

    python setup.py install


## Online Usage

```
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
```

The server for Realsense devices is started with `pyrs.Service()` which will printout the number of devices available. It can also be started as a context with `with pyrs.Service():`.

Different devices can be created from the service `Device` factory. They are created as their own class defined by device id, name, serial, firmware as well as enabled streams and camera presets. The default behaviour create a device with `id = 0` and setup the color, depth, pointcloud, color_aligned_depth, depth_aligned_color and infrared streams.

The available streams are either native or synthetic, and each one will create a property that exposes the current content of the frame buffer in the form of `device.<stream_name>`, where `<stream_name>` is color, depth, points, cad, dac or infrared. To get access to new data, `Device.wait_for_frames` has to be called once per frame.


## Offline Usage
```
## with connected device cam
from pyrealsense import offline
offline.save_depth_intrinsics(cam)
```

```
## previous device cam now offline
from pyrealsense import offline
offline.load_depth_intrinsics('610205001689')  # camera serial number
d = np.linspace(0, 1000, 480*640, dtype=np.uint16)
pc = offline.deproject_depth(d)
```

The module `offline` can store the rs_intrinsics and depth_scale of a device to disk by default in the user's home directory in the file `.pyrealsense`. This can later be loaded and used to deproject depth data into pointcloud, which is useful to store raw video file and save some disk memory.


## Examples

The wrapper comes with some [examples](https://github.com/toinsson/pyrealsense/tree/master/examples). You will need to install [matplotlib](http://matplotlib.org/), [opencv](http://opencv.org/) and [VTK](http://www.vtk.org/) to run everything. You can also look at the [Jupyter Notebook](https://github.com/toinsson/pyrealsense/blob/master/examples/show_jupyter.ipynb).


## Caveats

To this point, this wrapper has been tested with:

- [librealsense v1.12.1](https://github.com/IntelRealSense/librealsense/tree/v1.12.1)
- Ubuntu 16.04 LTS, Mac OS X 10.12.2 w/ SR300 camera
- Mac OS X 10.12.3 w/ R200 camera
- Windows 10

The offline module only supports a single camera.


## Build Status

Ubuntu Trusty, python 2 and 3: [![Build Status](https://travis-ci.org/toinsson/pyrealsense.svg?branch=master)](https://travis-ci.org/toinsson/pyrealsense)


## Possible Pull Requests

Contributions are always welcome. Make sure to push to the `dev` branch.

## Acknowledgements
This project has been developed as part of the E.U. Horizon 2020 research and innovation project Moregrasp (award number 643955).
