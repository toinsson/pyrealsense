import ctypes
from .constants import rs_stream, rs_format

class Stream(object):
    """Stream object that stores all necessary information for interaction with librealsense.

    Args:
        name (str): name of stream which will be used to create a ``@property`` on :func:`pyrealsense.core.Device`.
        native (bool): whether the stream is native or composite
        stream (str): from the parsed rs_stream
        width (int): name of the stream
        height (int): name of the stream
        format (int): from the parsed rs_format
        fps (int): name of the stream
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

class ColourStream(Stream):
    """Colour stream from device, with default parameters. See for possible combinations.
    """
    def __init__(self, name='color', width=640, height=480, fps=30):
        self.native = True
        self.stream = rs_stream.RS_STREAM_COLOR
        self.format = rs_format.RS_FORMAT_RGB8,
        self.shape = (height, width, 3)
        self.dtype = ctypes.c_uint8
        super(ColourStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)

class DepthStream(Stream):
    def __init__(self, name='depth', width=640, height=480, fps=30):
        self.native = True
        self.stream = rs_stream.RS_STREAM_DEPTH,
        self.format=rs_format.RS_FORMAT_Z16,
        self.shape = (height, width)
        self.dtype = ctypes.c_uint16
        super(DepthStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)

class PointStream(Stream):
    def __init__(self, name='points', width=640, height=480, fps=30):
        self.native = False,
        self.stream = rs_stream.RS_STREAM_POINTS,
        self.format = rs_format.RS_FORMAT_XYZ32F,
        self.shape = (height, width, 3)
        self.dtype = ctypes.c_float
        super(PointStream, self).__init__(name, self.native, self.stream, width, height, self.format, fps)

class CADStream(Stream):
    def __init__(self, name='cad',
                       native=False,
                       stream=rs_stream.RS_STREAM_COLOR_ALIGNED_TO_DEPTH,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_XYZ32F,
                       fps=30,
                       ):
        super(CADStream, self).__init__(name, native, stream, width, height, format, fps)
        self.shape = (height, width, 3)
        self.dtype = ctypes.c_uint8

class DACStream(Stream):
    def __init__(self, name='dac',
                       native=False,
                       stream=rs_stream.RS_STREAM_DEPTH_ALIGNED_TO_COLOR,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_XYZ32F,
                       fps=30,
                       ):
        super(DACStream, self).__init__(name, native, stream, width, height, format, fps)
        self.shape = (height, width)
        self.dtype = ctypes.c_uint16

class InfraredStream(Stream):
    def __init__(self, name='infrared',
                       native=True,
                       stream=rs_stream.RS_STREAM_INFRARED,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_Y8,
                       fps=30):
        super(InfraredStream, self).__init__(name, native, stream, width, height, format, fps)
        self.shape = (height, width)
        self.dtype = ctypes.c_uint8
