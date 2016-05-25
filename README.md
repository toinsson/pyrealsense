# pyrealsense
Simple python C extension to the [librealsense](https://github.com/IntelRealSense/librealsense) library. It allows to set configuration parameters on startup. It gives access to colour images, depth images and pointcloud as numpy arrays.

## installation

    python setup.py install

## usage

    import pyrealsense as pyrs
    pyrs.start()  # keyword arguments supported
    
    cm = pyrs.get_colour()
