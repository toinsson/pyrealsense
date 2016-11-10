import pyrealsense as pyrs
import matplotlib.pyplot as plt
pyrs.start()
dm = pyrs.get_depth()
plt.imshow(dm); plt.show()
pyrs.stop()
