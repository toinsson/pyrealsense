Examples
========

Matplotlib based
----------------

::

    import logging
    logging.basicConfig(level=logging.INFO)

    import matplotlib.pyplot as plt

    import pyrealsense as pyrs

    with pyrs.Service()
        dev = pyrs.Device()

        dev.wait_for_frame()
        plt.imshow(dev.colour)
        plt.show()


This example is using OpenCV.

And this one is based on VTK.