import logging
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
import pyrealsense as pyrs

with pyrs.Service() as serv:
    with serv.Device() as dev:
        dev.wait_for_frames()
        plt.imshow(dev.color)  # rgb by default
        plt.show()
