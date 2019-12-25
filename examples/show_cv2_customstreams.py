import logging
logging.basicConfig(level=logging.INFO)

import ctypes
import time
import numpy as np
import cv2
import pyrealsense as pyrs
from pyrealsense.constants import rs_option


class IRStreamR(pyrs.stream.Stream):
    def __init__(self, name='irl',
                 native=True,
                 stream=pyrs.constants.rs_stream.RS_STREAM_INFRARED,
                 width=640,
                 height=480,
                 format=pyrs.constants.rs_format.RS_FORMAT_Y8,
                 fps=30):
        super(IRStreamR, self).__init__(name, native, stream, width, height, format, fps)
        self.shape = (height, width, 1)
        self.dtype = ctypes.c_uint8


class IRStreamL(pyrs.stream.Stream):
    def __init__(self, name='irr',
                 native=True,
                 stream=pyrs.constants.rs_stream.RS_STREAM_INFRARED2,
                 width=640,
                 height=480,
                 format=pyrs.constants.rs_format.RS_FORMAT_Y8,
                 fps=30):
        super(IRStreamL, self).__init__(name, native, stream, width, height, format, fps)
        self.shape = (height, width, 1)
        self.dtype = ctypes.c_uint8


with pyrs.Service() as serv:
    with serv.Device(streams=(pyrs.stream.ColorStream(),
                              pyrs.stream.DepthStream(),
                              IRStreamR(),
                              IRStreamL(),
                              )) as dev:

        dev.apply_ivcam_preset(0)

        try:  # set auto exposure to obtain good depth image
            custom_options = [(rs_option.RS_OPTION_R200_LR_AUTO_EXPOSURE_ENABLED, True),
                              (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, True),
                             ]
            dev.set_device_options(*zip(*custom_options))
        except pyrs.RealsenseError:
            pass  # options are not available on all devices

        cnt = 0
        last = time.time()
        smoothing = 0.9
        fps_smooth = 30

        while True:

            cnt += 1
            if (cnt % 10) == 0:
                now = time.time()
                dt = now - last
                fps = 10/dt
                fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
                last = now

            dev.wait_for_frames()
            a = dev.irl
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
            b = dev.irr
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            c = dev.color
            c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
            d = dev.depth * dev.depth_scale * 1000
            d = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_RAINBOW)

            ab = np.concatenate((a,b), axis=1)
            cd = np.concatenate((c,d), axis=1)
            abcd = np.concatenate((ab,cd), axis=0)

            cv2.putText(abcd, str(fps_smooth)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (127, 127, 127))

            cv2.imshow('', abcd)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
