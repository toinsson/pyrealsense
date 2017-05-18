import ctypes


class rs_intrinsics(ctypes.Structure):
    """This is a 1-to-1 mapping to rs_intrinsics from librealsense.

    The `_fields_` class variable is defined as follows:

    * :attr:`width` (c_int): whatever
    * :attr:`height` (c_int): whatever
    * :attr:`ppx` (c_float): whatever
    * :attr:`ppy` (c_float): whatever
    * :attr:`fx` (c_float): whatever
    * :attr:`fy` (c_float): whatever
    * :attr:`model` (c_int): whatever
    * :attr:`coeffs` (c_float*5): whatever
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

    * :attr:`rotation` (c_float*9): whatever
    * :attr:`height` (c_float*3): whatever
    """
    _fields_ = [
    ("rotation", ctypes.c_float*9),  # column-major 3x3 rotation matrix
    ("translation", ctypes.c_float*3),  # element translation vector, in meters
    ]


class rs_error(ctypes.Structure):  # ERROR handling
    """This is a 1-to-1 mapping to rs_error from librealsense.

    The `_fields_` class variable is defined as follows:

    * :attr:`message` (c_char_p): whatever
    * :attr:`function` (pointer(c_char)): whatever
    * :attr:`args` (c_char_p): whatever
    """
    _fields_ = [("message", ctypes.c_char_p),
                ("function", ctypes.POINTER(ctypes.c_char)),
                ("args", ctypes.c_char_p),
                ]


class rs_context(ctypes.Structure):
    """This is a placeholder for the context. It is only defined to hold a reference to a void pointer.

    The `_fields_` class variable is defined as follows:

    * :attr:`body` (c_float): whatever
    """
    _fields_ = [("body", ctypes.c_float)]


class rs_device(ctypes.Structure):
    """This is a placeholder for the context. It is only defined to hold a reference to a void pointer.
    
    The `_fields_` class variable is defined as follows:

    * :attr:`body` (c_float): whatever
    """
    _fields_ = [("body", ctypes.c_float)]
