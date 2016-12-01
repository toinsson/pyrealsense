import sys

# import rsutil.so

from numpy.ctypeslib import ndpointer
import ctypes

from pyrealsense import constants as cnst
from pyrealsense.constants import rs_stream

## hack to load a manually exported rsutil.h
# import os
# _DIRNAME = os.path.dirname(__file__)
# rsutil = ctypes.CDLL(os.path.join(_DIRNAME,'rsutil.so'))

## import C lib
lrs = ctypes.CDLL('librealsense.so')


class rs_intrinsics(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("ppx", ctypes.c_float),
        ("ppy", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("model", ctypes.c_int),        #rs_distortion
        ("coeffs", ctypes.c_float*5),
    ]


## ERROR handling
# manual type definition
class rs_error(ctypes.Structure):
    # pass
    _fields_ = [("message", ctypes.c_char_p),
                ("function", ctypes.POINTER(ctypes.c_char)),
                ("args", ctypes.c_char_p),
                ]

e = ctypes.POINTER(rs_error)()

def _check_error():
    global e
    try:
        # TODO: currently problem with function argument of error, which SEGFAULT when accessed
        print("rs_error was raised with message: " + e.contents.message)
        sys.exit(0)
    except ValueError:
        # no error
        pass


ctx = 0
def start(device_id = 0):
    """Start a device with default parameters.
    """
    global ctx

    if not ctx:
        ctx = lrs.rs_create_context(cnst.RS_API_VERSION, ctypes.byref(e))
        _check_error()

    # print("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    lrs.rs_get_device_count(ctx, ctypes.byref(e))
    _check_error()
    dev = lrs.rs_get_device(ctx, 0, ctypes.byref(e))
    _check_error()


    charptr = ctypes.POINTER(ctypes.c_char)
    lrs.rs_get_device_name.restype=charptr

    ret = lrs.rs_get_device_name(dev, ctypes.byref(e))
    print "ret = ", ctypes.cast(ret, ctypes.c_char_p).value
    print("Using device 0, an %", ctypes.cast(ret, ctypes.c_char_p).value);
    _check_error();
    print("    Serial number: %s\n", lrs.rs_get_device_serial(dev, ctypes.byref(e)));
    _check_error();
    print("    Firmware version: %s\n", lrs.rs_get_device_firmware_version(dev, ctypes.byref(e)));
    _check_error();



    #"this should crash if there is no device.."

    #rs_enable_stream(dev, RS_STREAM_COLOR, c_width, c_height, RS_FORMAT_RGB8, c_fps, &e);
    lrs.rs_enable_stream(dev, rs_stream.RS_STREAM_DEPTH, 640, 480, 1, 30, ctypes.byref(e));
    _check_error();
    lrs.rs_enable_stream(dev, rs_stream.RS_STREAM_COLOR, 640, 480, 5, 30, ctypes.byref(e));
    _check_error();

    lrs.rs_start_device(dev, ctypes.byref(e))

    return Device(ctx, dev)

def stop():
    """Delete the context
    """
    global ctx
    lrs.rs_delete_context(ctx, ctypes.byref(e));



class Stream(object):
    """docstring for Stream"""
    def __init__(self, stream, width, height, format, fps):
        super(Stream, self).__init__()
        self.stream = stream
        self.width = width
        self.height = height
        self.format = format
        self.fps = fps

class ColourStream(Stream):
    def __init__(self, stream=cnst.rs_stream.RS_STREAM_COLOR,
                       width=640,
                       height=480,
                       format=cnst.rs_format.RS_FORMAT_RGB8,
                       fps=30):
        super(ColourStream, self).__init__(stream, width, height, format, fps)

class DepthStream(Stream):
    def __init__(self, stream=cnst.rs_stream.RS_STREAM_DEPTH,
                       width=640,
                       height=480,
                       format=cnst.rs_format.RS_FORMAT_Z16,
                       fps=30):
        super(DepthStream, self).__init__(stream, width, height, format, fps)

import rsutil

class Device(object):
    """docstring for device"""
    def __init__(self, ctx, dev, streams = []):
        super(Device, self).__init__()
        self.ctx = ctx
        self.dev = dev

        if streams == []:
            self.streams = [ColourStream(), DepthStream()]
        else:
            self.streams = streams

    def stop(self):
        """Stop a device  ##and delete the contexte
        """
        lrs.rs_stop_device(self.dev, ctypes.byref(e));


    def get_colour(self):
        """Return the color stream
        """
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint8, shape=(480,640,3))
        return lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_COLOR, ctypes.byref(e))


    def get_depth(self):
        """Return the depth stream
        """
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint16, shape=(480,640))
        return lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_DEPTH, ctypes.byref(e))


    def get_pointcloud(self):
        """Return the depth stream
        """
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint16, shape=(480,640))
        depth = lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_DEPTH, ctypes.byref(e))

        print depth.__class__

        ret = rsutil.pointcloud_from_depth(depth, 640, 480, 3)  # cython

        return ret

    def get_depth_scale(self):
        lrs.rs_get_device_depth_scale.restype = ctypes.c_float
        return lrs.rs_get_device_depth_scale(self.dev, ctypes.byref(e))

    def get_stream_intrinsics(self):

        _rs_intrinsics = rs_intrinsics()

        lrs.rs_get_stream_intrinsics(
            self.dev,
            rs_stream.RS_STREAM_COLOR,
            ctypes.byref(_rs_intrinsics),
            ctypes.byref(e))

        print _rs_intrinsics.width
        print [i for i in _rs_intrinsics.coeffs]

    def get_stream_intrinsics(self):

        _rs_intrinsics = rs_intrinsics()

        lrs.rs_get_stream_intrinsics(
            self.dev,
            rs_stream.RS_STREAM_DEPTH,
            ctypes.byref(_rs_intrinsics),
            ctypes.byref(e))

        print _rs_intrinsics.width
        print [i for i in _rs_intrinsics.coeffs]

        return _rs_intrinsics

    def test_intrinsics(self):

        _rs_intrinsics = rs_intrinsics()

        lrs.rs_get_stream_intrinsics(
            self.dev,
            rs_stream.RS_STREAM_DEPTH,
            ctypes.byref(_rs_intrinsics),
            ctypes.byref(e))

        print _rs_intrinsics.width
        print [i for i in _rs_intrinsics.coeffs]

        rsutil.test_intrinsics(_rs_intrinsics)

        return True