from unittest import TestCase
import pyrealsense as pyrs
from pyrealsense.utils import RealsenseError


class Test_Service_Device(TestCase):
    def test_is_started(self):
        try:
            service = pyrs.Service()
        except RealsenseError as e:
            self.assertTrue(e.function == 'rs_create_context')
        else:
            try:
                dev = service.Device()
            except RealsenseError as e:
                self.assertTrue(e.function == 'rs_get_device')
            else:
                dev.stop()
            finally:
                service.stop()
