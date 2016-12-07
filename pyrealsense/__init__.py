import sys

import logging

from numpy.ctypeslib import ndpointer
import ctypes

from pyrealsense import constants as cnst
from pyrealsense.constants import rs_stream, rs_format

# hack to load "extension" module
import os
_DIRNAME = os.path.dirname(__file__)
rsutil = ctypes.CDLL(os.path.join(_DIRNAME,'rsutilwrapper.so'))

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
class rs_error(ctypes.Structure):
    _fields_ = [("message", ctypes.c_char_p),
                ("function", ctypes.POINTER(ctypes.c_char)),
                ("args", ctypes.c_char_p),
                ]

def __pp(fun, *args):
    """Wrapper for printing char pointer from ctypes."""
    fun.restype = ctypes.POINTER(ctypes.c_char)
    ret = fun(*args)
    return ctypes.cast(ret, ctypes.c_char_p).value

e = ctypes.POINTER(rs_error)()

def _check_error():
    global e
    try:
        e.contents

        print("rs_error was raised when calling {}({})".format(
            __pp(lrs.rs_get_failed_function, e),
            __pp(lrs.rs_get_failed_args, e),
            ))
        print("    {}".format(__pp(lrs.rs_get_error_message, e)))
        sys.exit(0)

    except ValueError:
        # no error
        pass

ctx = 0
def start():
    """Start the service. Can only be one running.
    """
    global ctx

    device_id = 0

    if not ctx:
        ctx = lrs.rs_create_context(cnst.RS_API_VERSION, ctypes.byref(e))
        _check_error()

    print("There are {} connected RealSense devices.".format(
    lrs.rs_get_device_count(ctx, ctypes.byref(e))))
    _check_error()


def stop():
    """Stop the service
    """
    global ctx
    lrs.rs_delete_context(ctx, ctypes.byref(e));
    ctx = 0



class Stream(object):
    """docstring for Stream"""
    def __init__(self, name, stream, width, height, format, fps):
        super(Stream, self).__init__()
        self.name = name
        self.stream = stream
        self.width = width
        self.height = height
        self.format = format
        self.fps = fps

class ColourStream(Stream):
    def __init__(self, name='colour',
                       stream=rs_stream.RS_STREAM_COLOR,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_RGB8,
                       fps=30):
        super(ColourStream, self).__init__(name, stream, width, height, format, fps)

class DepthStream(Stream):
    def __init__(self, name='depth',
                       stream=rs_stream.RS_STREAM_DEPTH,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_Z16,
                       fps=30):
        super(DepthStream, self).__init__(name, stream, width, height, format, fps)


def pretty_print(fun, *args):
    """Wrapper for printing char pointer from ctypes."""
    fun.restype = ctypes.POINTER(ctypes.c_char)
    ret = fun(*args)
    return ctypes.cast(ret, ctypes.c_char_p).value


class Device(object):
    """Camera device."""
    def __init__(self, device_id = 0, streams = [ColourStream(), DepthStream()]):
        super(Device, self).__init__()

        global ctx

        self.dev = lrs.rs_get_device(ctx, device_id, ctypes.byref(e))
        _check_error()

        print("Using device {}, an {}".format(
            device_id, 
            pretty_print(lrs.rs_get_device_name, self.dev, ctypes.byref(e))))
        _check_error();
        print("    Serial number: {}".format(
            pretty_print(lrs.rs_get_device_serial, self.dev, ctypes.byref(e))))
        _check_error();
        print("    Firmware version: {}".format(
            pretty_print(lrs.rs_get_device_firmware_version, self.dev, ctypes.byref(e))))
        _check_error();

        self.streams = streams
        for s in self.streams:
            lrs.rs_enable_stream(self.dev,
                s.stream, s.width, s.height, s.format, s.fps,
                ctypes.byref(e));
            _check_error();

        lrs.rs_start_device(self.dev, ctypes.byref(e))

        for s in self.streams:
            print s.name
            _i = self._get_stream_intrinsics(s.format)
            setattr(self, s.name + '_intrinsics', _i)

            # if s.stream == 'depth'

        self._depth_scale = self._get_depth_scale()

    def stop(self):
        """Stop a device  ##and delete the contexte
        """
        lrs.rs_stop_device(self.dev, ctypes.byref(e));

    def wait_for_frame(self):
        """Block until new frames are available
        """
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))

    @property
    def colour(self):
        """Return the color stream
        """
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint8, shape=(480,640,3))
        return lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_COLOR, ctypes.byref(e))

    @property
    def depth(self):
        """Return the depth stream
        """
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint16, shape=(480,640))
        return lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_DEPTH, ctypes.byref(e))

    @property
    def pointcloud(self):
        """Return the depth stream
        """
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint16, shape=(480,640))
        depth = lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_DEPTH, ctypes.byref(e))

        ## ugly fix for outliers
        print depth[0,:2]
        depth[0,:2] = 0

        rsutil.get_pointcloud.restype = ndpointer(dtype=ctypes.c_float, shape=(480,640,3))
        return rsutil.get_pointcloud(
            ctypes.c_void_p(depth.ctypes.data),
            ctypes.byref(self.depth_intrinsics),
            ctypes.byref(ctypes.c_float(self.depth_scale)))

    @property
    def depth_scale(self):
        return self._depth_scale

    def _get_depth_scale(self):
        lrs.rs_get_device_depth_scale.restype = ctypes.c_float
        return lrs.rs_get_device_depth_scale(self.dev, ctypes.byref(e))

    def _get_stream_intrinsics(self, stream):
        _rs_intrinsics = rs_intrinsics()

        lrs.rs_get_stream_intrinsics(
            self.dev,
            stream,
            ctypes.byref(_rs_intrinsics),
            ctypes.byref(e))

        return _rs_intrinsics
