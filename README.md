# pyrealsense
Simple python C extension to the [librealsense](https://github.com/IntelRealSense/librealsense) library. 

It allows to set configuration parameters on startup via `ivcam_preset`. 

It returns colour images, depth images, pointcloud and uvmap as numpy arrays.

## installation

    python setup.py install

## usage

    import pyrealsense as pyrs
    pyrs.start()  # keyword arguments supported
    
    cm = pyrs.get_colour()

## caveats
To this point, this wrapper is only compatible with:
- Python 2.x
- Linux architecture
- SR300 camera

It is however simple to extend it further.
