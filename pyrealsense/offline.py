import os
from os import path
import yaml

from numpy.ctypeslib import ndpointer
import ctypes

from .to_wrap import rs_intrinsics


## hacky way to load "extension" module
_DIRNAME = os.path.dirname(__file__)
for file in os.listdir(_DIRNAME):
    if file.endswith(".so"):
        rsutilwrapper = file
rsutilwrapper = ctypes.CDLL(os.path.join(_DIRNAME, rsutilwrapper))


## global variable
depth_intrinsics = rs_intrinsics()
depth_scale = 0


def load_depth_intrinsics(dev_serial, fileloc = path.expanduser("~"), filename = '.pyrealsense'):
    global depth_intrinsics, depth_scale

    with open(path.join(fileloc, filename), 'r') as fh:
        d = yaml.load(fh)

    dev_d = d[dev_serial]
    for name, type_ in depth_intrinsics._fields_:
        obj = dev_d[name]
        if hasattr(obj, '__getitem__'):
            iter_obj = depth_intrinsics.__getattribute__(name)
            for i, o in enumerate(obj):
                iter_obj[i] = o
        else:
            depth_intrinsics.__setattr__(name, obj)

    depth_scale = dev_d['depth_scale']


def save_depth_intrinsics(dev, fileloc = path.expanduser("~"), filename = '.pyrealsense'):
    """Save intrinsics of camera for offline use, by default to 'home/.pyrealsense'."""

    ## TODO: - read first and update if needed
    #        - save other intrinsics

    intr = dev.__getattribute__('depth_intrinsics')
    dev_d = {dev.serial:{}}

    for name, type_ in intr._fields_:
        obj = intr.__getattribute__(name)

        if hasattr(obj, '__getitem__'):
            dev_d[dev.serial][name] = [i for i in obj]
        else:
            dev_d[dev.serial][name] = obj

    dev_d[dev.serial]['depth_scale'] = dev.depth_scale


    with open(path.join(fileloc, filename), 'w+') as fh:
        yaml.dump(dev_d, fh, default_flow_style=False, explicit_start=True)


def deproject_depth(depth):
    global depth_intrinsics, depth_scale

    width = depth_intrinsics.width
    height = depth_intrinsics.height

    rsutilwrapper.deproject_depth.restype = ndpointer(dtype=ctypes.c_float, shape=(height,width,3))

    return rsutilwrapper.deproject_depth(
        ctypes.c_void_p(depth.ctypes.data),
        ctypes.byref(depth_intrinsics),
        ctypes.byref(ctypes.c_float(depth_scale)))

