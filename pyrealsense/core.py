# -*- coding: utf-8 -*-
# Licensed under the Apache-2.0 License, see LICENSE for details.

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import six

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

from .constants import RS_API_VERSION, rs_stream, rs_option
from .stream import ColorStream, DepthStream, PointStream, CADStream, DACStream, InfraredStream
from .extstruct import rs_error, rs_intrinsics, rs_extrinsics, rs_context, rs_device
from .utils import pp, _check_error, RealsenseError, StreamMode, DeviceOptionRange
from .extlib import lrs, rsutilwrapper


class Service(object):
    """Context manager for librealsense service."""
    def __init__(self):
        super(Service, self).__init__()
        self.ctx = None
        self.start()

    def start(self):
        """Start librealsense service."""
        e = ctypes.POINTER(rs_error)()

        if not self.ctx:
            lrs.rs_create_context.restype = ctypes.POINTER(rs_context)
            self.ctx = lrs.rs_create_context(RS_API_VERSION, ctypes.byref(e))
            _check_error(e)

            # mirror librealsense behaviour of printing number of connected devices
            n_devices = lrs.rs_get_device_count(self.ctx, ctypes.byref(e))
            _check_error(e)
            logger.info('There are {} connected RealSense devices.'.format(n_devices))

    def stop(self):
        """Stop librealsense service."""
        if self.ctx:
            e = ctypes.POINTER(rs_error)()
            lrs.rs_delete_context(self.ctx, ctypes.byref(e))
            _check_error(e)
            self.ctx = None

    def get_devices(self):
        """Returns a generator that yields a dictionnary containing 'id', 'name', 'serial',
        'firmware' and 'is_streaming' keys.
        """
        e = ctypes.POINTER(rs_error)()
        n_devices = lrs.rs_get_device_count(self.ctx, ctypes.byref(e))
        _check_error(e)

        lrs.rs_get_device.restype = ctypes.POINTER(rs_device)
        for idx in range(n_devices):
            dev = lrs.rs_get_device(self.ctx, idx, ctypes.byref(e))
            _check_error(e)

            name = pp(lrs.rs_get_device_name, dev, ctypes.byref(e))
            _check_error(e)

            serial = pp(lrs.rs_get_device_serial, dev, ctypes.byref(e))
            _check_error(e)

            version = pp(lrs.rs_get_device_firmware_version, dev, ctypes.byref(e))
            _check_error(e)

            is_streaming = lrs.rs_is_device_streaming(dev, ctypes.byref(e))
            _check_error(e)

            yield {'id': idx, 'name': name, 'serial': serial,
                   'firmware': version, 'is_streaming': is_streaming}

    def get_device_modes(self, device_id):
        """Generates all different modes for the device which `id` is provided.

        Args:
            device_id (int): the device id as hinted by the output from :func:`start` or :func:`get_devices`.

        Returns: :obj:`generator` that yields all possible streaming modes as :obj:`StreamMode`.
        """
        e = ctypes.POINTER(rs_error)()
        dev = lrs.rs_get_device(self.ctx, device_id, ctypes.byref(e))
        _check_error(e)
        for stream_id in range(rs_stream.RS_STREAM_COUNT):
            mode_count = lrs.rs_get_stream_mode_count(dev, stream_id, ctypes.byref(e))
            _check_error(e)
            for idx in range(mode_count):
                width = ctypes.c_int()
                height = ctypes.c_int()
                fmt = ctypes.c_int()
                fps = ctypes.c_int()
                lrs.rs_get_stream_mode(dev, stream_id, idx,
                                       ctypes.byref(width),
                                       ctypes.byref(height),
                                       ctypes.byref(fmt),
                                       ctypes.byref(fps),
                                       ctypes.byref(e))
                _check_error(e)
                yield StreamMode(stream_id, width.value, height.value,
                                 fmt.value, fps.value)

    def is_device_streaming(self, device_id):
        """Indicates if device is streaming.

        Utility function which does not require to enumerate all devices
        or to initialize a Device object.
        """
        e = ctypes.POINTER(rs_error)()
        lrs.rs_get_device.restype = ctypes.POINTER(rs_device)
        dev = lrs.rs_get_device(self.ctx, device_id, ctypes.byref(e))
        _check_error(e)
        is_streaming = lrs.rs_is_device_streaming(dev, ctypes.byref(e))
        _check_error(e)
        return is_streaming

    def Device(self, *args, **kwargs):
        """Factory function which returns a :obj:`Device`, also accepts optionnal arguments.
        """
        return Device(self, *args, **kwargs)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        # print 'stopping the service'
        self.stop()

    def __del__(self):
        self.stop()

    def __nonzero__(self):
        return bool(self.ctx)

    def __bool__(self):
        return bool(self.ctx)


def Device(service, device_id=0, streams=None, depth_control_preset=None, ivcam_preset=None):
    """Camera device, which subclass :class:`DeviceBase` and create properties for each input
    streams to expose their data. It should be instantiated through :func:`Service.Device`.

    Args:
        service (:obj:`Service`): any running service.
        device_id (int): the device id as hinted by the output from :func:`start`.
        streams (:obj:`list` of :obj:`pyrealsense.stream.Stream`): if None, all streams will be
            enabled with their default parameters (e.g `640x480@30FPS`)
        depth_control_preset (int): optional preset to be applied.
        ivcam_preset (int): optional preset to be applied with input value from
            :obj:`pyrealsense.constants.rs_ivcam_preset`.
    Returns:
        A subclass of :class:`DeviceBase` which class name includes the device serial number.
    """
    e = ctypes.POINTER(rs_error)()

    assert service.ctx, 'Service needs to be started'
    ctx = service.ctx

    if streams is None:
        streams = [ColorStream(), DepthStream(), PointStream(), CADStream(), DACStream(), InfraredStream()]

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

    # create a new class for the device
    class_name = name.split(" ")[-1] + "-" + serial
    NewDevice = type(class_name, (DeviceBase,), dict())

    nd = NewDevice(dev, device_id, name, serial, version, streams)
    if nd.is_streaming():
        # Device is already running.
        # It is not possible to enable further streams.
        return nd

    # enable the stream and start device
    for s in streams:
        if s.native:
            lrs.rs_enable_stream(dev, s.stream, s.width, s.height, s.format, s.fps, ctypes.byref(e))
            _check_error(e)

    lrs.rs_start_device(dev, ctypes.byref(e))

    # depth control preset
    if depth_control_preset:
        rsutilwrapper.apply_depth_control_preset(dev, depth_control_preset)

    # ivcam preset
    if ivcam_preset:
        rsutilwrapper.apply_ivcam_preset(dev, ivcam_preset)

    # add stream property and intrinsics
    for s in streams:
        if s.native:
            setattr(NewDevice, s.name + '_intrinsics', nd._get_stream_intrinsics(s.stream))
        setattr(NewDevice, s.name, property(nd._get_stream_data_closure(s)))

    # add manually depth_scale and manual pointcloud
    for s in streams:
        if s.name == 'depth':
            setattr(NewDevice, 'depth_scale', property(lambda x: nd._get_depth_scale()))
        if s.name == 'points':
            setattr(NewDevice, 'pointcloud', property(lambda x: nd._get_pointcloud()))

    return nd


class DeviceBase(object):
    """Camera device base class which is called via the :func:`Device` factory. It
        exposes the main functions from librealsense.
    """
    def __init__(self, dev, device_id, name, serial, version, streams):
        super(DeviceBase, self).__init__()
        assert dev, 'Device was not initialized correctly'

        self.dev = dev
        self.device_id = device_id
        self.name = name
        self.serial = serial
        self.version = version
        self.streams = streams

    def stop(self):
        """End data acquisition.
        Raises:
            :class:`utils.RealsenseError`: in case librealsense reports a problem.
        """
        if self.dev and self.is_streaming():
            e = ctypes.POINTER(rs_error)()
            lrs.rs_stop_device(self.dev, ctypes.byref(e))
            _check_error(e)
        self.dev = None

    def is_streaming(self):
        """Indicates if device is streaming.

        Returns:
            (bool): return value of `lrs.rs_is_device_streaming`.
        """
        if self.dev:
            e = ctypes.POINTER(rs_error)()
            is_streaming = lrs.rs_is_device_streaming(self.dev, ctypes.byref(e))
            _check_error(e)
            return bool(is_streaming)
        else:
            return False

    def poll_for_frame(self):
        """Check if new frames are available, without blocking.

        Returns:
            int: 1 if new frames are available, 0 if no new frames have arrived

        Raises:
            :class:`utils.RealsenseError`: in case librealsense reports a problem.
        """
        e = ctypes.POINTER(rs_error)()
        res = lrs.rs_poll_for_frames(self.dev, ctypes.byref(e))
        _check_error(e)
        return res

    def wait_for_frames(self):
        """Block until new frames are available.

        Raises:
            :class:`utils.RealsenseError`: in case librealsense reports a problem.
        """
        e = ctypes.POINTER(rs_error)()
        lrs.rs_wait_for_frames(self.dev, ctypes.byref(e))
        _check_error(e)

    def get_frame_timestamp(self, stream):
        """Retrieve the time at which the latest frame on a specific stream was captured.

        Args:
            stream (int): stream id

        Returns:
            (long): timestamp
        """
        lrs.rs_get_frame_timestamp.restype = ctypes.c_double
        e = ctypes.POINTER(rs_error)()
        return lrs.rs_get_frame_timestamp(self.dev, stream, ctypes.byref(e))

    def get_frame_number(self, stream):
        """Retrieve the frame number for specific stream.

        Args:
            stream (int): value from :class:`pyrealsense.constants.rs_stream`.

        Returns:
            (double): frame number.
        """
        lrs.rs_get_frame_number.restype = ctypes.c_ulonglong
        e = ctypes.POINTER(rs_error)()
        return lrs.rs_get_frame_number(self.dev, stream, ctypes.byref(e))

    def get_device_extrinsics(self, from_stream, to_stream):
        """Retrieve extrinsic transformation between the viewpoints of two different streams.

        Args:
            from_stream (:class:`pyrealsense.constants.rs_stream`): from stream.
            to_stream (:class:`pyrealsense.constants.rs_stream`): to stream.

        Returns:
            (:class:`pyrealsense.extstruct.rs_extrinsics`): extrinsics parameters as a structure

        """
        e = ctypes.POINTER(rs_error)()
        _rs_extrinsics = rs_extrinsics()
        lrs.rs_get_device_extrinsics(
            self.dev,
            from_stream,
            to_stream,
            ctypes.byref(_rs_extrinsics),
            ctypes.byref(e))
        _check_error(e)
        return _rs_extrinsics

    def get_device_modes(self):
        """Returns a generator that yields all possible streaming modes as :obj:`StreamMode`."""
        e = ctypes.POINTER(rs_error)()
        for stream in self.streams:
            mode_count = lrs.rs_get_stream_mode_count(self.dev, stream.stream, ctypes.byref(e))
            _check_error(e)
            for idx in range(mode_count):
                width = ctypes.c_int()
                height = ctypes.c_int()
                fmt = ctypes.c_int()
                fps = ctypes.c_int()
                lrs.rs_get_stream_mode(self.dev, stream.stream, idx,
                                       ctypes.byref(width),
                                       ctypes.byref(height),
                                       ctypes.byref(fmt),
                                       ctypes.byref(fps),
                                       ctypes.byref(e))
                _check_error(e)
                yield StreamMode(stream.stream, width.value, height.value,
                                 fmt.value, fps.value)

    def get_available_options(self):
        """Returns available options as a list of (:obj:`DeviceOptionRange`, value).
        """
        avail_opt_ranges = []
        for option in range(rs_option.RS_OPTION_COUNT):
            try:
                opt_range = self.get_device_option_range_ex(option)
            except RealsenseError:
                pass
            else:
                avail_opt_ranges.append(opt_range)

        avail_opt = [r.option for r in avail_opt_ranges]
        return six.moves.zip(avail_opt_ranges, self.get_device_options(avail_opt))

    def get_device_options(self, options):
        """Get device options.

        Args:
            option (:obj:`list` of int): taken from :class:`pyrealsense.constants.rs_option`.

        Returns:
            (:obj:`iter` of double): options values.
        """
        e = ctypes.POINTER(rs_error)()
        current_values = (ctypes.c_double*len(options))()
        option_array_type = ctypes.c_int*len(options)
        option_array = option_array_type(*options)
        lrs.rs_get_device_options.argtypes = [ctypes.POINTER(rs_device),
                                              option_array_type,
                                              ctypes.c_int,
                                              ctypes.POINTER(ctypes.c_double),
                                              ctypes.POINTER(ctypes.POINTER(rs_error))]
        lrs.rs_get_device_options.restype = None
        lrs.rs_get_device_options(self.dev, option_array, len(options), current_values, ctypes.byref(e))
        _check_error(e)
        return iter(current_values)

    def set_device_options(self, options, values):
        """Set device options.

        Args:
            option (:obj:`list` of int): taken from :class:`pyrealsense.constants.rs_option`.

            values (:obj:`list` of double): options values.
        """
        assert len(options) == len(values)
        e = ctypes.POINTER(rs_error)()
        count = len(options)
        option_array_type = ctypes.c_int * count
        values_array_type = ctypes.c_double * count

        lrs.rs_set_device_options.argtypes = [ctypes.POINTER(rs_device),
                                              option_array_type,
                                              ctypes.c_int,
                                              values_array_type,
                                              ctypes.POINTER(ctypes.POINTER(rs_error))]
        lrs.rs_set_device_options.restype = None
        c_options = option_array_type(*options)
        c_values = values_array_type(*values)
        lrs.rs_set_device_options(self.dev, c_options, count, c_values, ctypes.byref(e))
        _check_error(e)

    def get_device_option(self, option):
        """Get device option.

        Args:
            option (int): taken from :class:`pyrealsense.constants.rs_option`.

        Returns:
            (double): option value.
        """
        lrs.rs_get_device_option.restype = ctypes.c_double
        e = ctypes.POINTER(rs_error)()
        return lrs.rs_get_device_option(self.dev, option, ctypes.byref(e))

    def set_device_option(self, option, value):
        """Set device option.

        Args:
            option (int): taken from :class:`pyrealsense.constants.rs_option`.
            value (double): value to be set for the option.
        """
        e = ctypes.POINTER(rs_error)()
        lrs.rs_set_device_option(self.dev, ctypes.c_uint(option), ctypes.c_double(value), ctypes.byref(e))
        _check_error(e)

    def get_device_option_range_ex(self, option):
        """Get device option range.

        Args:
            option (int): taken from :class:`pyrealsense.constants.rs_option`.

        Returns:
            (:obj:`DeviceOptionRange`): option range.
        """
        e = ctypes.POINTER(rs_error)()
        min_ = ctypes.c_double()
        max_ = ctypes.c_double()
        step = ctypes.c_double()
        defv = ctypes.c_double()
        lrs.rs_get_device_option_range_ex(self.dev, option, ctypes.byref(min_),
                                          ctypes.byref(max_), ctypes.byref(step),
                                          ctypes.byref(defv), ctypes.byref(e))
        _check_error(e)
        return DeviceOptionRange(option, min_.value, max_.value, step.value, defv.value)

    def get_device_option_description(self, option):
        """Get the device option description.

        Args:
            option (int): taken from :class:`pyrealsense.constants.rs_option`.

        Returns:
            (str): option value.
        """
        e = ctypes.POINTER(rs_error)()
        return pp(lrs.rs_get_device_option_description, self.dev, ctypes.c_uint(option), ctypes.byref(e))

    def reset_device_options_to_default(self, options):
        """Reset device options to default.

        Args:
            option (:obj:`list` of int): taken from :class:`pyrealsense.constants.rs_option`.
        """
        e = ctypes.POINTER(rs_error)()
        count = len(options)
        option_array_type = ctypes.c_int * count
        lrs.rs_reset_device_options_to_default.argtypes = [ctypes.POINTER(rs_device),
                                                           option_array_type,
                                                           ctypes.c_int,
                                                           ctypes.POINTER(ctypes.POINTER(rs_error))]
        lrs.rs_reset_device_options_to_default.restype = None
        c_options = option_array_type(*options)
        lrs.rs_reset_device_options_to_default(self.dev, c_options, count, ctypes.byref(e))
        _check_error(e)

    def _get_stream_intrinsics(self, stream):
        e = ctypes.POINTER(rs_error)()
        _rs_intrinsics = rs_intrinsics()
        lrs.rs_get_stream_intrinsics(
            self.dev,
            stream,
            ctypes.byref(_rs_intrinsics),
            ctypes.byref(e))
        return _rs_intrinsics

    def _get_stream_data_closure(self, s):
        def get_stream_data(s):
            e = ctypes.POINTER(rs_error)()
            lrs.rs_get_frame_data.restype = ndpointer(dtype=s.dtype, shape=s.shape)
            try:
                data = lrs.rs_get_frame_data(self.dev, s.stream, ctypes.byref(e))
            except TypeError:
                _check_error(e)
                raise
            else:
                return data
        return lambda x: get_stream_data(s)

    def _get_depth_scale(self):
        e = ctypes.POINTER(rs_error)()
        lrs.rs_get_device_depth_scale.restype = ctypes.c_float
        return lrs.rs_get_device_depth_scale(self.dev, ctypes.byref(e))

    def _get_pointcloud(self):
        ds = [s for s in self.streams if type(s) is DepthStream][0]

        e = ctypes.POINTER(rs_error)()
        lrs.rs_get_frame_data.restype = ndpointer(dtype=ctypes.c_uint16, shape=(ds.height, ds.width))
        depth = lrs.rs_get_frame_data(self.dev, rs_stream.RS_STREAM_DEPTH, ctypes.byref(e))

        pointcloud = np.zeros((ds.height * ds.width * 3), dtype=np.float32)

        # ugly fix for outliers
        depth[0, :2] = 0

        rsutilwrapper.deproject_depth(pointcloud, self.depth_intrinsics, depth, self.depth_scale)
        return pointcloud.reshape((ds.height, ds.width, 3))

    def apply_ivcam_preset(self, preset):
        """Provide access to several recommend sets of option presets for ivcam.

        Args:
            preset (int): preset from (:obj:`pyrealsense.constants.rs_ivcam_preset`)

        """
        rsutilwrapper.apply_ivcam_preset(self.dev, preset)

    def project_point_to_pixel(self, point):
        """Project a 3d point to its 2d pixel coordinate by calling rsutil's
        rs_project_point_to_pixel under the hood.

        Args:
            point (np.array): (x,y,z) coordinate of the point

        Returns:
            pixel (np.array): (x,y) coordinate of the pixel
        """
        pixel = np.ones(2, dtype=np.float32) * np.NaN
        rsutilwrapper.project_point_to_pixel(pixel, self.depth_intrinsics, point)
        return pixel

    def deproject_pixel_to_point(self, pixel, depth):
        """Deproject a 2d pixel to its 3d point coordinate by calling rsutil's
        rs_deproject_pixel_to_point under the hood.

        Args:
            pixel (np.array): (x,y) coordinate of the point
            depth (float): depth at that pixel

        Returns:
            point (np.array): (x,y,z) coordinate of the point
        """
        point = np.ones(3, dtype=np.float32) * np.NaN
        rsutilwrapper.deproject_pixel_to_point(point, self.depth_intrinsics, pixel, depth)
        return point

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()

    def __str__(self):
        return '{}(serial={}, firmware={})'.format(self.__class__.__name__,
                                                   self.serial, self.firmware)

    def __nonzero__(self):
        return self.is_streaming()

    def __bool__(self):
        return self.is_streaming()

