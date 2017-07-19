from unittest import TestCase
import pyrealsense as pyrs
import numpy as np


class Test_0_Start(TestCase):
    def test_is_started(self):
        service = pyrs.Service()
        ret = len(list(service.get_devices()))
        self.assertTrue(ret > 0)
        service.stop()


class Test_1_Device(TestCase):
    def test_is_not_created(self):
        service = pyrs.Service()
        cam = service.Device()
        self.assertTrue(cam.is_streaming())
        cam.wait_for_frames()
        self.assertTrue(cam.color.any())
        cam.stop()
        service.stop()


class Test_2_Wrapper(TestCase):
    def test_is_wrapped(self):
        service = pyrs.Service()
        cam = service.Device()
        self.assertTrue(cam.is_streaming())
        cam.wait_for_frames()

        from pyrealsense import rsutilwrapper

        pc = cam.points
        dm = cam.depth

        nz = np.nonzero(pc)
        x0, y0 = nz[0][0], nz[1][0]

        pixel0 = np.ones(2, dtype=np.float32) * np.NaN
        point0 = pc[x0, y0].astype(np.float32)

        rsutilwrapper.project_point_to_pixel(pixel0, cam.depth_intrinsics, point0)

        point1 = np.zeros(3, dtype=np.float32)
        x1, y1 = np.round(pixel0[1]).astype(int), np.round(pixel0[0]).astype(int)

        self.assertTrue(np.isclose([x0, y0], [x1, y1], atol=2).all())

        depth = dm[x0, y0] * cam.depth_scale
        rsutilwrapper.deproject_pixel_to_point(point1, cam.depth_intrinsics, pixel0, depth)

        self.assertTrue(np.isclose(point0, point1, atol=10e-3).all())
        cam.stop()
        service.stop()
