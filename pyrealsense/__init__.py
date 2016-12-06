import sys

from numpy.ctypeslib import ndpointer
import ctypes

from pyrealsense import constants as cnst
from pyrealsense.constants import rs_stream, rs_format

# hack to load "extension"
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
def start(device_id = 0):
    """Start a device with default parameters.
    """
    global ctx

    if not ctx:
        ctx = lrs.rs_create_context(cnst.RS_API_VERSION, ctypes.byref(e))
        _check_error()

    print("There are {} connected RealSense devices.".format(
    lrs.rs_get_device_count(ctx, ctypes.byref(e))))
    _check_error()
    dev = lrs.rs_get_device(ctx, 0, ctypes.byref(e))
    _check_error()

    print("Using device 0, an {}".format(__pp(lrs.rs_get_device_name, dev, ctypes.byref(e))))
    _check_error();
    print("    Serial number: {}".format(__pp(lrs.rs_get_device_serial, dev, ctypes.byref(e))))
    _check_error();
    print("    Firmware version: {}".format(
            __pp(lrs.rs_get_device_firmware_version, dev, ctypes.byref(e))))
    _check_error();

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
    def __init__(self, stream=rs_stream.RS_STREAM_COLOR,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_RGB8,
                       fps=30):
        super(ColourStream, self).__init__(stream, width, height, format, fps)

class DepthStream(Stream):
    def __init__(self, stream=rs_stream.RS_STREAM_DEPTH,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_Z16,
                       fps=30):
        super(DepthStream, self).__init__(stream, width, height, format, fps)


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

        self.depth_intrinsics = self.get_stream_intrinsics()
        self.depth_scale = self.get_depth_scale()



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


        # depth_intrinsics = self.get_stream_intrinsics()
        # depth_scale = self.get_depth_scale()

        # import ipdb; ipdb.set_trace()

        # print depth.__class__
        # print depth.shape
        # print depth.dtype
        # print depth.ctypes
        # print depth_intrinsics.__class__
        # print depth_scale.__class__

        # # for i in range(10):
        # print [depth.reshape(-1)[i] for i in range(10)]
        # # for i in range(200,240):
        # print [depth.reshape(-1)[i] for i in range(200,240)]

        rsutil.get_pointcloud.restype = ndpointer(dtype=ctypes.c_float, shape=(480,640,3))

        ret = rsutil.get_pointcloud(
            ctypes.c_void_p(depth.ctypes.data),
            ctypes.byref(self.depth_intrinsics),
            ctypes.byref(ctypes.c_float(self.depth_scale)))

        return ret

    def get_depth_scale(self):
        lrs.rs_get_device_depth_scale.restype = ctypes.c_float
        return lrs.rs_get_device_depth_scale(self.dev, ctypes.byref(e))

    def get_stream_intrinsics(self):

        _rs_intrinsics = rs_intrinsics()

        lrs.rs_get_stream_intrinsics(
            self.dev,
            rs_stream.RS_STREAM_DEPTH,
            ctypes.byref(_rs_intrinsics),
            ctypes.byref(e))

        # print _rs_intrinsics.width
        # print [i for i in _rs_intrinsics.coeffs]

        return _rs_intrinsics
