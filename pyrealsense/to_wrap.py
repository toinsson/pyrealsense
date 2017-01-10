import ctypes

class rs_intrinsics(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("ppx", ctypes.c_float),
        ("ppy", ctypes.c_float),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("model", ctypes.c_int),        #rs_distortion
        ("coeffs", ctypes.c_float*5),
    ]


## ERROR handling
class rs_error(ctypes.Structure):
    _fields_ = [("message", ctypes.c_char_p),
                ("function", ctypes.POINTER(ctypes.c_char)),
                ("args", ctypes.c_char_p),
                ]

## Mockup for context class
class rs_context(ctypes.Structure):
    _fields_ = [("body", ctypes.c_float)]

class rs_device(ctypes.Structure):
    _fields_ = [("body", ctypes.c_float)]

