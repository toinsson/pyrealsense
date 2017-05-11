import ctypes
from .constants import rs_stream, rs_format

class Stream(object):
    """Stream object that stores all necessary information for interaction with librealsense.
    
    :param string name: name of the stream
    :param bool native: whether the stream is native or composite
    :param string stream: from the parsed rs_stream
    :param int width: name of the stream
    :param int height: name of the stream
    :param int format: from the parsed rs_format
    :param int fps: name of the stream
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
    def __init__(self, width=640, height=480, fps=30):
        self.name = 'colour'
        self.native = True
        self.stream = rs_stream.RS_STREAM_COLOR
        self.format = rs_format.RS_FORMAT_RGB8,

        self.shape = (height, width, 3)
        self.dtype = ctypes.c_uint8

        super(ColourStream, self).__init__(name, native, stream, width, height, format, fps)

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

