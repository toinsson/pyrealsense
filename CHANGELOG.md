2.2.0
=====

Major:
- Add support for YUV color streams. Replace `ColorStream` parameter `use_bgr` with `color_format`. `color_format` can be one of `'rgb'`, `'bgr'`, `'yuv'`.

Minor:
- Add jupyter example
- Add depth-only example including histogram normalization

2.1.0
=====

Major:
- Replace global context with single context per service instance
- Add `Device` factory function to `core.Service`

Minor:
- Add `utils.StreamMode` to wrap stream modes
- Add `DeviceOptionRange` to wrap device option ranges
- Add `core.Device.get_device_option_range_ex`: Mirrors librealsense, returns `DeviceOptionRange`
- Add `core.Device.get_device_options`: Mirrors librealsense, returns list of doubles
- Add `core.Device.get_available_options`: Uses `get_device_option_range_ex` to find all available options and queries their current value efficiently using `get_device_options`
- Add `core.Device.set_device_options()` to set multipe options at once, mirrors librealsense

2.0.0
=====

Complete refactor by @toinsson
