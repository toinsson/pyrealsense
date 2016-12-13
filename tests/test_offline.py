from unittest import TestCase
import pyrealsense as pyrs

class TestStart(TestCase):
    def test_is_started(self):
        try:
            pyrs.start()
        except pyrs.RealsenseError as e:
            self.assertTrue(e.function == 'rs_create_context')

class TestDevice(TestCase):
    def test_is_not_created(self):
        try:
            pyrs.Device()
        except pyrs.RealsenseError as e:
            self.assertTrue(e.function == 'rs_get_device')
