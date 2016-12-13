from unittest import TestCase
import pyrealsense as pyrs

class Test_0_Start(TestCase):
    def test_is_started(self):
        ret = pyrs.start()
        self.assertTrue(ret == 1)


class Test_1_Device(TestCase):
    def test_is_not_created(self):
        cam = pyrs.Device()
        cam.wait_for_frame()
        c = cam.colour
        self.assertTrue(cam.colour.any())