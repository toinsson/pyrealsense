import pyrealsense as pyrs
import matplotlib.pyplot as plt
dev = pyrs.start()
dm = dev.get_depth()
plt.imshow(dm); plt.show()
pyrs.stop()
