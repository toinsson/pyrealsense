import pyrealsense as pyrs
import matplotlib.pyplot as plt
dev = pyrs.start()
dm = dev.get_depth()
plt.imshow(dm); plt.show()

print dev.get_depth_scale()
pc = dev.get_pointcloud()

cm = dev.get_colour()
plt.imshow(cm); plt.show()

# import ipdb; ipdb.set_trace()


from sys import path as  sys_path
sys_path.insert(1,'/home/antoine/Documents/owndev/fistwriter')
from utils import vtk_plot
vtk_plot.plot_objects(pc, axis=True)

print '1'

pyrs.stop()

print '2'

# d = np.array(range(3), dtype=np.float32)
# rsutil.print_array(d.ctypes, d.ctypes)




# import cv2
# # import pyrealsense as pyds
# # pyds.start()
# # time.sleep(2)

# while True:
#     frame = dev.get_colour()

#     cv2.imshow('', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

