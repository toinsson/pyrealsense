

cdef extern from "librealsense/rsutil.h":
    ctypedef struct rs_intrinsics:
        pass

    void rs_deproject_pixel_to_point(float point[3], rs_intrinsics * intrin, float pixel[2], float depth)


def pointcloudfromdepth(point, intrin, pixel, depth):
    pass

#     memset(pointcloud, 0, sizeof(pointcloud));

#     int dx, dy;
#     for(dy=0; dy<depth_intrin.height; ++dy)
#     {
#         for(dx=0; dx<depth_intrin.width; ++dx)
#         {
#             uint16_t depth_value = depth_image[dy * depth_intrin.width + dx];
#             float depth_in_meters = depth_value * scale;

#             if(depth_value == 0) continue;

#             float depth_pixel[2] = {(float)dx, (float)dy};
#             float depth_point[3];

#             rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth_in_meters);

#             pointcloud[dy*depth_intrin.width*3 + dx*3 + 0] = depth_point[0];
#             pointcloud[dy*depth_intrin.width*3 + dx*3 + 1] = depth_point[1];
#             pointcloud[dy*depth_intrin.width*3 + dx*3 + 2] = depth_point[2];
#         }
#     }

#     npy_intp dims[3] = {depth_intrin.height, depth_intrin.width, 3};

#     return PyArray_SimpleNewFromData(
#         3,
#         dims,
#         NPY_FLOAT,
#         (void*) &pointcloud
#         );
# }


