import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

import ctypes
from .extlib import lrs

class RealsenseError(Exception):
    """Error thrown during the processing in case the processing chain needs to be exited. 
    Will printout the error message as received from librealsense."""
    def __init__(self, function, args, message):
        self.function = function
        self.args = args
        self.message = message

    def __str__(self):
        args = "".join([c for c in self.args])
        return "{}({}) crashed with: {}".format(self.function, args, self.message)


def _check_error(e):
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


def pp(fun, *args):
    """Wrapper for printing char pointer from ctypes."""
    fun.restype = ctypes.POINTER(ctypes.c_char)
    ret = fun(*args)
    val = ctypes.cast(ret, ctypes.c_char_p).value

    ## Python 2/3 difference
    if type(val) == str: return val
    if type(val) == bytes: return val.decode("utf-8")

