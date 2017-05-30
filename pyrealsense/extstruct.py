import ctypes


class rs_intrinsics(ctypes.Structure):
    """This is a 1-to-1 mapping to rs_intrinsics from librealsense.

    The `_fields_` class variable is defined as follows:

    * :attr:`width` (c_int): width of the image in pixels
    * :attr:`height` (c_int): height of the image in pixels
    * :attr:`ppx` (c_float): horizontal coordinate of the principal point of the image, as a pixel offset from the left edge
    * :attr:`ppy` (c_float): vertical coordinate of the principal point of the image, as a pixel offset from the top edge
    * :attr:`fx` (c_float): focal length of the image plane, as a multiple of pixel width
    * :attr:`fy` (c_float): focal length of the image plane, as a multiple of pixel height
    * :attr:`model` (c_int): distortion model of the image
    * :attr:`coeffs` (c_float*5): distortion coefficients
    """
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("ppx", ctypes.c_float),
        ("ppy", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("model", ctypes.c_int),        # rs_distortion
        ("coeffs", ctypes.c_float*5),
    ]


class rs_extrinsics(ctypes.Structure):
    """This is a 1-to-1 mapping to rs_extrinsics from librealsense.

    The `_fields_` class variable is defined as follows:

    * :attr:`rotation` (c_float*9): column-major 3x3 rotation matrix
    * :attr:`height` (c_float*3): 3 element translation vector, in meters
    """
    _fields_ = [
    ("rotation", ctypes.c_float*9),
    ("translation", ctypes.c_float*3),
    ]


class rs_error(ctypes.Structure):
    """This is a 1-to-1 mapping to rs_error from librealsense.

    The `_fields_` class variable is defined as follows:

    * :attr:`message` (c_char_p): error message
    * :attr:`function` (pointer(c_char)): function which caused the error
    * :attr:`args` (c_char_p): arguments to the function which caused the error
    """
    _fields_ = [("message", ctypes.c_char_p),
                ("function", ctypes.POINTER(ctypes.c_char)),
                ("args", ctypes.c_char_p),
                ]


class rs_context(ctypes.Structure):
    """This is a placeholder for the context. It is only defined to hold a reference to a pointer.
    """
    pass

class rs_device(ctypes.Structure):
    """This is a placeholder for the context. It is only defined to hold a reference to a pointer.
    """
    pass

