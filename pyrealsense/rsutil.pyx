import numpy as np
cimport numpy as np


import ctypes

# class ct_rs_intrinsics(ctypes.Structure):
#     _fields_ = [
#         ("width", ctypes.c_int),
#         ("height", ctypes.c_int),
#         ("ppx", ctypes.c_float),
#         ("ppy", ctypes.c_float),
#         ("fx", ctypes.c_float),
#         ("fy", ctypes.c_float),
#         ("model", ctypes.c_int),        #rs_distortion
#         ("coeffs", ctypes.c_float*5),
#     ]


ctypedef enum rs_distortion:
    RS_DISTORTION_NONE                  
    RS_DISTORTION_MODIFIED_BROWN_CONRADY
    RS_DISTORTION_INVERSE_BROWN_CONRADY 
    RS_DISTORTION_FTHETA                
    RS_DISTORTION_COUNT

ctypedef struct rs_intrinsics:
    int           width
    int           height
    float         ppx
    float         ppy
    float         fx
    float         fy
    rs_distortion model
    float         coeffs[5]


cdef extern from "librealsense/rsutil.h":
    # ctypedef struct rs_intrinsics:
    #     pass

    void rs_deproject_pixel_to_point(float point[3], rs_intrinsics * intrin, float pixel[2], float depth)


cdef float pointcloud[480*640*3];

def test_intrinsics(ct_rs_intrinsics):
    print ct_rs_intrinsics
    cdef float depth_point[3]
    cdef float depth_pixel[2]
    cdef float depth_in_meters
    cdef rs_intrinsics ri

    ri.width = ct_rs_intrinsics.ri

    ret = rs_deproject_pixel_to_point(
        depth_point,
        &ri,
        depth_pixel,
        depth_in_meters)

    

    return ret

def pointcloud_from_depth(np.ndarray depth_image, width, height, scale):
    print "from cython"

    # memset(pointcloud, 0, sizeof(pointcloud));

    cdef int dx, dy

    cdef int depth_value
    cdef float depth_in_meters
    cdef float depth_pixel[2]
    cdef float depth_point[3]

    cdef rs_intrinsics _rs_intrinsics
    # for(dy=0; dy<height; ++dy)
    # {
    for dy in range(height):
    #     for(dx=0; dx<width; ++dx)
    #     {
        for dx in range(width):

            depth_value = depth_image[dy,dx]
            depth_in_meters = depth_value * scale

            if(depth_value == 0): continue

            depth_pixel[0] = <float>dx
            depth_pixel[1] = <float>dy
            depth_point[3]

            rs_deproject_pixel_to_point(depth_point, &_rs_intrinsics, depth_pixel, depth_in_meters)

            pointcloud[dy*width*3 + dx*3 + 0] = depth_point[0]
            pointcloud[dy*width*3 + dx*3 + 1] = depth_point[1]
            pointcloud[dy*width*3 + dx*3 + 2] = depth_point[2]
    #     }
    # }

    return pointcloud
    # npy_intp dims[3] = {depth_intrin.height, depth_intrin.width, 3};
    # return PyArray_SimpleNewFromData(
    #     3,
    #     dims,
    #     NPY_FLOAT,
    #     <void*> &pointcloud
    #     );


