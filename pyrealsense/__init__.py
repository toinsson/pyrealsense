import sys

from numpy.ctypeslib import ndpointer
import ctypes

from pyrealsense import constants as cnst
from pyrealsense.constants import rs_stream

# manual type definition
class rs_error(ctypes.Structure):
    _fields_ = [("message", ctypes.c_char_p),
                ("function", ctypes.c_char_p),
                ("args", ctypes.c_char_p),
                ]


# typedef struct rs_intrinsics
# {
#     int           width;     /* width of the image in pixels */
#     int           height;    /* height of the image in pixels */
#     float         ppx;       /* horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
#     float         ppy;       /* vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
#     float         fx;        /* focal length of the image plane, as a multiple of pixel width */
#     float         fy;        /* focal length of the image plane, as a multiple of pixel height */
#     rs_distortion model;     /* distortion model of the image */
#     float         coeffs[5]; /* distortion coefficients */
# } rs_intrinsics;

# class rs_intrinsics(ctypes.Structure):
#     _fields_ = [("rotation"), ctypes.float[9]
#                 ]


# typedef struct rs_extrinsics
# {
#     float rotation[9];    /* column-major 3x3 rotation matrix */
#     float translation[3]; /* 3 element translation vector, in meters */
# } rs_extrinsics;


## import C lib
lrs = ctypes.CDLL('librealsense.so')


## ERROR handling
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


class device(object):
    """docstring for device"""
    def __init__(self, ctx, dev):
        super(device, self).__init__()
        self.ctx = ctx
        self.dev = dev


    def stop():
        """Stop a device  ##and delete the contexte
        """
        lrs.rs_stop_device(self.dev, ctypes.byref(e));


    def get_colour():
        """Return the color stream
        """
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint8, shape=(480,640,3))
        return lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_COLOR, ctypes.byref(e))


    def get_depth():
        """Return the depth stream
        """
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint16, shape=(480,640))
        return lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_DEPTH, ctypes.byref(e))


ctx = 0
def start(device_id = 0):
    """Start a device with default parameters.
    """
    global ctx

    if not ctx:
        ctx = lrs.rs_create_context(cnst.RS_API_VERSION, ctypes.byref(e))
        _check_error()

    lrs.rs_get_device_count(ctx, ctypes.byref(e))
    _check_error()

    dev = lrs.rs_get_device(ctx, 0, ctypes.byref(e))
    _check_error()

    #"this should crash if there is no device.."

    #rs_enable_stream(dev, RS_STREAM_COLOR, c_width, c_height, RS_FORMAT_RGB8, c_fps, &e);
    lrs.rs_enable_stream(dev, rs_stream.RS_STREAM_DEPTH, 640, 480, 1, 30, ctypes.byref(e));
    _check_error();
    lrs.rs_enable_stream(dev, rs_stream.RS_STREAM_COLOR, 640, 480, 5, 30, ctypes.byref(e));
    _check_error();

    lrs.rs_start_device(dev, ctypes.byref(e))

    return device(ctx, dev)

def stop():
    """Delete the context
    """
    global ctx
    lrs.rs_delete_context(self.ctx, ctypes.byref(e));
