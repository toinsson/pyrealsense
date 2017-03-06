import ctypes
from .constants import rs_stream, rs_format

class Stream(object):
    """docstring for Stream"""
    def __init__(self, name, native, stream, width, height, format, fps):
        super(Stream, self).__init__()
        self.name = name
        self.native = native
        self.stream = stream
        self.width = width
        self.height = height
        self.format = format
        self.fps = fps

class ColourStream(Stream):
    def __init__(self, name='colour',
                       native=True,
                       stream=rs_stream.RS_STREAM_COLOR,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_RGB8,
                       fps=30):
        super(ColourStream, self).__init__(name, native, stream, width, height, format, fps)
        self.shape = (height, width, 3)
        self.dtype = ctypes.c_uint8

class DepthStream(Stream):
    def __init__(self, name='depth',
                       native=True,
                       stream=rs_stream.RS_STREAM_DEPTH,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_Z16,
                       fps=30):
        super(DepthStream, self).__init__(name, native, stream, width, height, format, fps)
        self.shape = (height, width)
        self.dtype = ctypes.c_uint16

class PointStream(Stream):
    def __init__(self, name='points',
                       native=False,
                       stream=rs_stream.RS_STREAM_POINTS,
                       width=640,
                       height=480,
                       format=rs_format.RS_FORMAT_XYZ32F,
                       fps=30):
        super(PointStream, self).__init__(name, native, stream, width, height, format, fps)
        self.shape = (height, width, 3)
        self.dtype = ctypes.c_float

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

