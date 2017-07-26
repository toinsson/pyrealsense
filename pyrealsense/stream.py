# -*- coding: utf-8 -*-
# Licensed under the Apache-2.0 License, see LICENSE for details.

import ctypes
from .constants import rs_stream, rs_format


class Stream(object):
    """Stream object that stores all necessary information for interaction with librealsense.
    See for possible combinations.

    Args:
        name (str): name of stream which will be used to create a ``@property`` on :class:`pyrealsense.core.DeviceBase`.
        native (bool): whether the stream is native or composite
        stream (int): from :class:`pyrealsense.constants.rs_stream`
        width (int): width
        height (int): height
        format (int): from :class:`pyrealsense.constants.rs_format`
        fps (int): fps
    """
    def __init__(self, name, native, stream, width, height, format, fps):
        super(Stream, self).__init__()
        self.name = name
        self.native = native
        self.stream = stream
        self.format = format
        self.width = width
        self.height = height
        self.fps = fps


class ColorStream(Stream):
    """Color stream from device, with default parameters.
    """
    def __init__(self, name='color', width=640, height=480, fps=30, color_format='rgb'):
        self.native = True
        self.stream = rs_stream.RS_STREAM_COLOR
        if color_format == 'rgb':
            self.format = rs_format.RS_FORMAT_RGB8
            n_channels = 3
        elif color_format == 'bgr':
            self.format = rs_format.RS_FORMAT_BGR8
            n_channels = 3
        elif color_format == 'yuv':
            self.format = rs_format.RS_FORMAT_YUYV
            n_channels = 2
        else:
            raise ValueError('Unknown color format. Expected rgb, bgr, or yuv ({} given)').format(color_format)
        self.shape = (height, width, n_channels)
        self.dtype = ctypes.c_uint8
        super(ColorStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)


class DepthStream(Stream):
    """Depth stream from device, with default parameters.
    """
    def __init__(self, name='depth', width=640, height=480, fps=30):
        self.native = True
        self.stream = rs_stream.RS_STREAM_DEPTH
        self.format = rs_format.RS_FORMAT_Z16
        self.shape = (height, width)
        self.dtype = ctypes.c_uint16
        super(DepthStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)


class PointStream(Stream):
    """Point stream from device, with default parameters.
    """
    def __init__(self, name='points', width=640, height=480, fps=30):
        self.native = False
        self.stream = rs_stream.RS_STREAM_POINTS
        self.format = rs_format.RS_FORMAT_XYZ32F
        self.shape = (height, width, 3)
        self.dtype = ctypes.c_float
        super(PointStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)


class CADStream(Stream):
    """CAD stream from device, with default parameters.
    """
    def __init__(self, name='cad', width=640, height=480, fps=30, color_format='rgb'):
        self.native = False
        self.stream = rs_stream.RS_STREAM_COLOR_ALIGNED_TO_DEPTH
        if color_format == 'rgb':
            self.format = rs_format.RS_FORMAT_RGB8
            n_channels = 3
        elif color_format == 'bgr':
            self.format = rs_format.RS_FORMAT_BGR8
            n_channels = 3
        elif color_format == 'yuv':
            self.format = rs_format.RS_FORMAT_YUYV
            n_channels = 2
        else:
            raise ValueError('Unknown color format. Expected rgb, bgr, or yuv ({} given)').format(color_format)
        self.shape = (height, width, n_channels)
        self.dtype = ctypes.c_uint8
        super(CADStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)


class DACStream(Stream):
    """DAC stream from device, with default parameters.
    """
    def __init__(self, name='dac', width=640, height=480, fps=30):
        self.native = False
        self.stream = rs_stream.RS_STREAM_DEPTH_ALIGNED_TO_COLOR
        self.format = rs_format.RS_FORMAT_Z16
        self.shape = (height, width)
        self.dtype = ctypes.c_uint16
        super(DACStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)


class InfraredStream(Stream):
    """Infrared stream from device, with default parameters.
    """
    def __init__(self, name='infrared', width=640, height=480, fps=30):
        self.native = True
        self.stream = rs_stream.RS_STREAM_INFRARED
        self.format = rs_format.RS_FORMAT_Y8
        self.shape = (height, width)
        self.dtype = ctypes.c_uint8
        super(InfraredStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)
