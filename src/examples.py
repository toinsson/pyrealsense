import time

import cv2
import pyrealsense as pyrs
pyrs.start()
time.sleep(2)

while True:
    frame = pyrs.get_depth_map()

    cv2.imshow('', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break