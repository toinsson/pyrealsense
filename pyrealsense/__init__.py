import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import ctypes
from numpy.ctypeslib import ndpointer
from .constants import RS_API_VERSION, rs_stream, rs_format
from .stream import ColourStream, DepthStream, PointStream, CADStream, DACStream
from .to_wrap import rs_error, rs_intrinsics, rs_extrinsics, rs_context, rs_device
from .utils import pp, _check_error
from .importlib import rsutilwrapper, lrs

# Global variables
e = ctypes.POINTER(rs_error)()
ctx = 0


def start():
    """Start the service. Can only be one running."""
    global ctx, e

    # if not ctx:
    lrs.rs_create_context.restype = ctypes.POINTER(rs_context)
    ctx = lrs.rs_create_context(RS_API_VERSION, ctypes.byref(e))
    _check_error(e)

    n_devices = lrs.rs_get_device_count(ctx, ctypes.byref(e))
    logger.info("There are {} connected RealSense devices.".format(n_devices))
    _check_error(e)

    return n_devices


def stop():
    """Stop the service."""
    global ctx, e

    lrs.rs_delete_context(ctx, ctypes.byref(e))
    _check_error(e)
    ctx = 0


def Device(device_id=0, streams=None, depth_control_preset=None, ivcam_preset=None):
    """Camera device."""

    global ctx, e

    if streams is None:
        streams = [ColourStream(), DepthStream(), PointStream(), CADStream(), DACStream()]

    lrs.rs_get_device.restype = ctypes.POINTER(rs_device)
    dev = lrs.rs_get_device(ctx, device_id, ctypes.byref(e))
    _check_error(e)
    name = pp(lrs.rs_get_device_name, dev, ctypes.byref(e))
    _check_error(e)
    serial = pp(lrs.rs_get_device_serial, dev, ctypes.byref(e))
    _check_error(e)
    version = pp(lrs.rs_get_device_firmware_version, dev, ctypes.byref(e))
    _check_error(e)

    logger.info("Using device {}, an {}".format(device_id, name))
    logger.info("    Serial number: {}".format(serial))
    logger.info("    Firmware version: {}".format(version))

    ## create a new class for the device
    class_name = name.split(" ")[-1] + "-" + serial
    NewDevice = type(class_name, (DeviceBase,), dict())

    nd = NewDevice(dev, name, serial, version, streams)

    ## enable the stream and start device
    for s in streams:
        if s.native:
            lrs.rs_enable_stream(dev,
                s.stream, s.width, s.height, s.format, s.fps,
                ctypes.byref(e));
            _check_error(e);

    lrs.rs_start_device(dev, ctypes.byref(e))

    ## depth control preset
    if depth_control_preset:
        rsutilwrapper._apply_depth_control_preset(dev, depth_control_preset)

    ## ivcam preset
    if ivcam_preset:
        rsutilwrapper._apply_ivcam_preset(dev, ivcam_preset)

    ## add stream property and intrinsics
    for s in streams:
        if s.native:
            setattr(NewDevice, s.name + '_intrinsics', nd._get_stream_intrinsics(s.stream))
        setattr(NewDevice, s.name, property(nd._get_stream_data_closure(s)))

    ## add manually depth_scale and manual pointcloud
    for s in streams:
        if s.name == 'depth':
            setattr(NewDevice, 'depth_scale', property(lambda x: nd._get_depth_scale()))
        if s.name == 'points':
            setattr(NewDevice, 'pointcloud', property(lambda x: nd._get_pointcloud()))

    return nd


class DeviceBase(object):
    """Camera device base class."""
    def __init__(self, dev, name, serial, version, streams):
        super(DeviceBase, self).__init__()
        global ctx, e

        self.dev = dev
        self.name = name
        self.serial = serial
        self.version = version
        self.streams = streams

    def stop(self):
        """Stop a device."""
        lrs.rs_stop_device(self.dev, ctypes.byref(e));

    def poll_for_frame(self):
        """Block until new frames are available."""
        res = lrs.rs_poll_for_frames(self.dev, ctypes.byref(e))
        _check_error(e)
        return res

    def wait_for_frame(self):
        """Block until new frames are available."""
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))
        _check_error(e)

    def get_frame_timestamp(self, stream):
        """Get the frame number"""
        lrs.rs_get_frame_timestamp.restype = ctypes.c_double
        return lrs.rs_get_frame_timestamp(self.dev, stream, ctypes.byref(e))

    def get_frame_number(self, stream):
        """Get the frame number"""
        lrs.rs_get_frame_number.restype = ctypes.c_ulonglong
        return lrs.rs_get_frame_number(self.dev, stream, ctypes.byref(e))

    def get_device_extrinsics(self, from_stream, to_stream):
        """Retrieve extrinsic transformation between the viewpoints of two different streams."""
        _rs_extrinsics = rs_extrinsics()
        lrs.rs_get_device_extrinsics(
            self.dev,
            from_stream,
            to_stream,
            ctypes.byref(_rs_extrinsics),
            ctypes.byref(e))
        _check_error(e)
        return _rs_extrinsics

    def get_device_option(self, option):
        """Get device option."""
        lrs.rs_get_device_option.restype = ctypes.c_double
        return lrs.rs_get_device_option(self.dev, option, ctypes.byref(e))

    def get_device_option_description(self, option):
        """Get the device option description."""
        return pp(lrs.rs_get_device_option_description, 
            self.dev, ctypes.c_uint(option), ctypes.byref(e))

    def set_device_option(self, option, value):
        """Set device option."""
        lrs.rs_set_device_option(self.dev,
            ctypes.c_uint(option), ctypes.c_double(value), ctypes.byref(e))
        _check_error(e)

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

        rsutilwrapper.deproject_depth.restype = ndpointer(dtype=ctypes.c_float, shape=(480,640,3))

        return rsutilwrapper.deproject_depth(
            ctypes.c_void_p(depth.ctypes.data),
            ctypes.byref(self.depth_intrinsics),
            ctypes.byref(ctypes.c_float(self.depth_scale)))

    def project_point_to_pixel(self, point):
        ## make sure we have float32
        point = point.astype(ctypes.c_float)

        rsutilwrapper.project_point_to_pixel.restype = ndpointer(dtype=ctypes.c_float, shape=(2,))

        return rsutilwrapper.project_point_to_pixel(
            ctypes.c_void_p(point.ctypes.data),
            ctypes.byref(self.depth_intrinsics),
            )

    def deproject_pixel_to_point(self, pixel, depth):
        ## make sure we have float32
        pixel = pixel.astype(ctypes.c_float)
        depth = depth.astype(ctypes.c_float)

        rsutilwrapper.deproject_pixel_to_point.restype = ndpointer(dtype=ctypes.c_float, shape=(3,))

        return rsutilwrapper.deproject_pixel_to_point(
            ctypes.c_void_p(pixel.ctypes.data),
            ctypes.c_float(depth),
            ctypes.byref(self.depth_intrinsics),
            )
