import pyrealsense as pyrs
import matplotlib.pyplot as plt

pyrs.start()
dev = pyrs.Device()

dev.wait_for_frame()
plt.imshow(dev.colour)
plt.show()
