import numpy as np
cimport numpy as np

import ctypes


cdef extern from "rs.h":
    cdef struct rs_device:
        pass
    cdef struct rs_error:
        pass
    cdef enum rs_ivcam_preset:
        pass
    cdef struct rs_intrinsics:
        pass


cdef extern from "rsutilwrapper.h":
    int fc( int N, double* a, double* b, double* z )  # z = a + b

    void _apply_depth_control_preset(rs_device* device, int preset)
    void _apply_ivcam_preset(rs_device* device, rs_ivcam_preset preset)
    void _project_point_to_pixel(float* pixel, const rs_intrinsics * intrin, const float* point)
    void _deproject_pixel_to_point(float* point, const rs_intrinsics * intrin, const float* pixel, float depth)


from libc.stdint cimport uintptr_t

def apply_depth_control_preset(device, preset):
    cdef uintptr_t adr = <uintptr_t>ctypes.addressof(device.contents)
    _apply_ivcam_preset(<rs_device*>adr, preset)

def apply_ivcam_preset(device, preset):
    cdef uintptr_t adr = <uintptr_t>ctypes.addressof(device.contents)
    _apply_ivcam_preset(<rs_device*>adr, <rs_ivcam_preset>preset)

def project_point_to_pixel(np.ndarray pixel, intrin, np.ndarray point):
    cdef uintptr_t adr = <uintptr_t>ctypes.addressof(intrin)
    _project_point_to_pixel(<float*> pixel.data, <rs_intrinsics*>adr, <float*> point.data)

def deproject_pixel_to_point(np.ndarray point, intrin, np.ndarray pixel, depth):
    cdef uintptr_t adr = <uintptr_t>ctypes.addressof(intrin)
    _deproject_pixel_to_point(<float*> point.data, <rs_intrinsics*>adr, <float*> pixel.data, depth)


def fpy( N,
    np.ndarray[np.double_t,ndim=1] A,
    np.ndarray[np.double_t,ndim=1] B,
    np.ndarray[np.double_t,ndim=1] Z ):
    """ wrap np arrays to fc( a.data ... ) """
    assert N <= len(A) == len(B) == len(Z)
    fcret = fc( N, <double*> A.data, <double*> B.data, <double*> Z.data )
        # fcret = fc( N, A.data, B.data, Z.data )  grr char*
    return fcret

def translate_by_value(rs_intrinsics_py, depth):
    print rs_intrinsics_py.__class__, depth
    cdef rs_intrinsics rs_intrinsics_cy
    rs_intrinsics_cy.width = rs_intrinsics_py.width
    print rs_intrinsics_cy.width

# import ctypes
# cpdef test(rs_device_ct, depth):

#     print rs_device_ct, depth
#     # cdef rs_device* rs_device_cy
#     # rs_device_cy = ctypes.address(rs_device_ct)
#     # cdef int depth
#     # depth = 1
#     cdef long adr = <long>ctypes.addressof(rs_device_ct.contents)

#     cdef rs_error* rse

#     _apply_depth_control_preset(<rs_device*>adr, depth)

#     res = rs_get_device_name(<rs_device*>adr, &rse)

#     print res

# void _apply_depth_control_preset(rs_device * device, int preset)
# {
#     rs_apply_depth_control_preset(device, preset);
# }

