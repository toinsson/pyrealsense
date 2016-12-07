# pyrealsense
Simple ctypes extension to the [librealsense](https://github.com/IntelRealSense/librealsense) library. 

## installation

    python setup.py install

## usage

    import pyrealsense as pyrs
    from pyrs import
    pyrs.start()  # setup the context

    cam = pyrs.Device(device_id = 0, streams=[pyrs.ColourStream(fps=60))

    cam.wait_for_frames()
    print(cam.colour)

## caveats
To this point, this wrapper has only been tested with:
- Python 2.x
- Linux architecture
- SR300 camera

It is however simple to extend it further.
