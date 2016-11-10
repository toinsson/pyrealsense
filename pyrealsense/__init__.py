import sys

from numpy.ctypeslib import ndpointer
import ctypes


# type definition

class rs_error(ctypes.Structure):
    _fields_ = [("message", ctypes.c_char_p),
                ("function", ctypes.c_char_p),
                ("args", ctypes.c_char_p),
                ]


## parse that on build
RS_API_MAJOR_VERSION = 1
RS_API_MINOR_VERSION = 9
RS_API_PATCH_VERSION = 7
RS_API_VERSION = (
    ((RS_API_MAJOR_VERSION) * 10000) + ((RS_API_MINOR_VERSION) * 100) + (RS_API_PATCH_VERSION))


## import C lib

lrs = ctypes.CDLL('librealsense.so')


## ERROR handling
e = ctypes.POINTER(rs_error)()

def _check_error():
    try:
        # TODO: currently problem with function argument of error, which SEGFAULT when accessed
        print("rs_error was raised with message: " + e.contents.message)
        sys.exit(0)
    except ValueError:
        # no error
        pass

dev = 0

def start():
    """Start a device with default parameters.
    """
    global dev

    ctx = lrs.rs_create_context(RS_API_VERSION, ctypes.byref(e))
    _check_error()

    lrs.rs_get_device_count(ctx, ctypes.byref(e))
    _check_error()

    dev = lrs.rs_get_device(ctx, 0, ctypes.byref(e))
    _check_error()

    #rs_enable_stream(dev, RS_STREAM_COLOR, c_width, c_height, RS_FORMAT_RGB8, c_fps, &e);
    lrs.rs_enable_stream(dev, 1, 640, 480, 5, 30, ctypes.byref(e));

    lrs.rs_start_device(dev, ctypes.byref(e))

def get_colour():
    """Return
    """
    global dev

    lrs.rs_wait_for_frames(dev, ctypes.byref(e))
    lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint8, shape=(480,640,3))
    return lrs.rs_get_frame_data(dev, 1, ctypes.byref(e))
