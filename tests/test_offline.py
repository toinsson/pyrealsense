from unittest import TestCase
import pyrealsense as pyrs
from pyrealsense.utils import RealsenseError

class Test_0_Start(TestCase):
    def test_is_started(self):
        try:
            pyrs.start()
        except RealsenseError as e:
            self.assertTrue(e.function == b'rs_create_context')

class Test_1_Device(TestCase):
    def test_is_not_created(self):
        try:
            pyrs.Device()
        except RealsenseError as e:
            self.assertTrue(e.function == b'rs_get_device')
