import sys

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.ctypeslib import ndpointer
import ctypes

from pyrealsense.constants import RS_API_VERSION, rs_stream, rs_format
from pyrealsense.stream import ColourStream, DepthStream, PointStream, CADStream
from pyrealsense.to_wrap import rs_error, rs_intrinsics


# hack to load "extension" module
import os
_DIRNAME = os.path.dirname(__file__)
rsutilwrapper = ctypes.CDLL(os.path.join(_DIRNAME,'rsutilwrapper.so'))

## import C lib
lrs = ctypes.CDLL('librealsense.so')


def pp(fun, *args):
    """Wrapper for printing char pointer from ctypes."""
    fun.restype = ctypes.POINTER(ctypes.c_char)
    ret = fun(*args)
    return ctypes.cast(ret, ctypes.c_char_p).value

e = ctypes.POINTER(rs_error)()


class RealsenseError(Exception):
    """Error thrown during the processing in case the processing chain needs to be exited."""
    def __init__(self, function, args, message):
        self.function = function
        self.args = args
        self.message = message

    def __str__(self):
        args = "".join([c for c in self.args])
        return "{}({}) crashed with: {}".format(self.function, args, self.message)


def _check_error():
    global e
    try:
        e.contents

        logger.error("rs_error was raised when calling {}({})".format(
            pp(lrs.rs_get_failed_function, e),
            pp(lrs.rs_get_failed_args, e),
            ))
        logger.error("    {}".format(pp(lrs.rs_get_error_message, e)))

        raise RealsenseError(pp(lrs.rs_get_failed_function, e),
                pp(lrs.rs_get_failed_args, e),
                pp(lrs.rs_get_error_message, e))

    except ValueError:
        # no error
        pass

ctx = 0
def start():
    """Start the service. Can only be one running."""
    global ctx

    if not ctx:
        ctx = lrs.rs_create_context(RS_API_VERSION, ctypes.byref(e))
        _check_error()

    n_devices = lrs.rs_get_device_count(ctx, ctypes.byref(e))
    logger.info("There are {} connected RealSense devices.".format(n_devices))
    _check_error()

    return n_devices


def stop():
    """Stop the service."""
    global ctx
    lrs.rs_delete_context(ctx, ctypes.byref(e));
    ctx = 0


class Device(object):
    """Camera device."""
    def __init__(self,
            device_id = 0,
            streams = [ColourStream(), DepthStream(), PointStream(), CADStream()],
            depth_control_preset = None,
            ivcam_preset = None):
        super(Device, self).__init__()

        global ctx

        self.dev = lrs.rs_get_device(ctx, device_id, ctypes.byref(e))
        _check_error()

        logger.info("Using device {}, an {}".format(
            device_id, 
            pp(lrs.rs_get_device_name, self.dev, ctypes.byref(e))))
        _check_error();
        logger.info("    Serial number: {}".format(
            pp(lrs.rs_get_device_serial, self.dev, ctypes.byref(e))))
        _check_error();
        logger.info("    Firmware version: {}".format(
            pp(lrs.rs_get_device_firmware_version, self.dev, ctypes.byref(e))))
        _check_error();

        ## depth control preset
        if depth_control_preset:
            rsutilwrapper._apply_depth_control_preset(self.dev, depth_control_preset)
        ## ivcam preset
        if ivcam_preset:
            rsutilwrapper._apply_ivcam_preset(self.dev, ivcam_preset)

        self.streams = streams
        for s in self.streams:
            if s.native:
                lrs.rs_enable_stream(self.dev,
                    s.stream, s.width, s.height, s.format, s.fps,
                    ctypes.byref(e));
                _check_error();

        lrs.rs_start_device(self.dev, ctypes.byref(e))

        ## add stream property and intrinsics
        for s in self.streams:
            if s.native:
                setattr(self, s.name + '_intrinsics', self._get_stream_intrinsics(s.stream))

            setattr(Device, s.name, property(self._get_stream_data_closure(s)))

        ## add manually depth_scale and manual pointcloud
        for s in self.streams:
            if s.name == 'depth':
                setattr(Device, 'depth_scale', property(lambda x: self._get_depth_scale()))

            if s.name == 'points':
                setattr(Device, 'pointcloud', property(lambda x: self._get_pointcloud()))

    def stop(self):
        """Stop a device."""
        lrs.rs_stop_device(self.dev, ctypes.byref(e));

    def wait_for_frame(self):
        """Block until new frames are available."""
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))

    def _get_stream_intrinsics(self, stream):
        _rs_intrinsics = rs_intrinsics()
        lrs.rs_get_stream_intrinsics(
            self.dev,
            stream,
            ctypes.byref(_rs_intrinsics),
            ctypes.byref(e))
        return _rs_intrinsics

    def _get_stream_data_closure(self, s):
        def get_stream_data(s):
            lrs.rs_get_frame_data.restype = ndpointer(dtype=s.dtype, shape=s.shape)
            return lrs.rs_get_frame_data(self.dev, s.stream, ctypes.byref(e))
        return lambda x: get_stream_data(s)

    def _get_depth_scale(self):
        lrs.rs_get_device_depth_scale.restype = ctypes.c_float
        return lrs.rs_get_device_depth_scale(self.dev, ctypes.byref(e))

    def _get_pointcloud(self):
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint16, shape=(480,640))
        depth = lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_DEPTH, ctypes.byref(e))

        ## ugly fix for outliers 
        depth[0,:2] = 0

        rsutilwrapper.get_pointcloud.restype = ndpointer(dtype=ctypes.c_float, shape=(480,640,3))

        return rsutilwrapper.get_pointcloud(
            ctypes.c_void_p(depth.ctypes.data),
            ctypes.byref(self.depth_intrinsics),
            ctypes.byref(ctypes.c_float(self.depth_scale)))

