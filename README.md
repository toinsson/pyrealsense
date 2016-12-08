# pyrealsense
Simple ctypes extension to the [librealsense](https://github.com/IntelRealSense/librealsense) library. 

## installation

    python setup.py install

## usage

    import pyrealsense as pyrs
    pyrs.start()  # setup the context
    cam = pyrs.Device(device_id = 0, streams=[pyrs.ColourStream(fps=60)])
    cam.wait_for_frame()
    print(cam.colour)

The server for Realsense devices is started with:

    pyrs.start()

which will printout the number of devices available.

Different devices can be created from the `Device` class. They are defined by device id and streams passed on creation. The default behaviour create a device with `id = 0` and setup the colour, depth, pointcloud and colour_aligned_depth streams.

The available streams are either native or synthetic, and each one will create a property on the device object that exposes the current content of the frame buffer in the form of `device.stream_name`, where `stream_name` can be colour, depth, points, cad or dac. To get access to new data, `Device.wait_for_frame` has to be called.

## caveats
To this point, this wrapper has only been tested with:
- Python 2.x
- Linux architecture
- SR300 camera

