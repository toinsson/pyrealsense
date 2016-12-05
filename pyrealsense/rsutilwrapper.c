#include "rs.h"
#include "assert.h"

#include <stdio.h>

void rs_project_point_to_pixel(float pixel[2], const struct rs_intrinsics * intrin, const float point[3])
{
    assert(intrin->model != RS_DISTORTION_INVERSE_BROWN_CONRADY); // Cannot project to an inverse-distorted image
    assert(intrin->model != RS_DISTORTION_FTHETA); // Cannot project to an ftheta image

    float x = point[0] / point[2], y = point[1] / point[2];
    if(intrin->model == RS_DISTORTION_MODIFIED_BROWN_CONRADY)
    {
        float r2  = x*x + y*y;
        float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
        x *= f;
        y *= f;
        float dx = x + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
        float dy = y + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
        x = dx;
        y = dy;
    }
    pixel[0] = x * intrin->fx + intrin->ppx;
    pixel[1] = y * intrin->fy + intrin->ppy;
}

void rs_deproject_pixel_to_point(float point[3], const struct rs_intrinsics * intrin, const float pixel[2], float depth)
{
    assert(intrin->model != RS_DISTORTION_MODIFIED_BROWN_CONRADY); // Cannot deproject from a forward-distorted image
    assert(intrin->model != RS_DISTORTION_FTHETA); // Cannot deproject to an ftheta image

    float x = (pixel[0] - intrin->ppx) / intrin->fx;
    float y = (pixel[1] - intrin->ppy) / intrin->fy;
    if(intrin->model == RS_DISTORTION_INVERSE_BROWN_CONRADY)
    {
        float r2  = x*x + y*y;
        float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
        float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
        float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
        x = ux;
        y = uy;
    }
    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;
}

void print_array(float point_[3], float* point__)
{
    for (int i = 0; i < 3; i++)
    {
        printf("%f ", point_[i]);
        printf("%f ", point__[i]);
    }
}

// float pointcloud_[480*640*3];

// static PyObject *pointcloud_from_depth(PyObject *self, PyObject *args)
// {
//     PyArrayObject *depth_p = NULL;
//     if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &depth_p))
//         return NULL;

//     memset(pointcloud_, 0, sizeof(pointcloud_));

//     int dx, dy;
//     for(dy=0; dy<depth_intrin.height; ++dy)
//     {
//         for(dx=0; dx<depth_intrin.width; ++dx)
//         {
//             /* Retrieve the 16-bit depth value and map it into a depth in meters */
//             uint16_t depth_value = ((uint16_t*) depth_p->data)[dy * depth_intrin.width + dx];
//             float depth_in_meters = depth_value * scale;

//             /* Skip over pixels with a depth value of zero, which is used to indicate no data */
//             if(depth_value == 0) continue;

//             /* Map from pixel coordinates in the depth image to pixel coordinates in the color image  */
//             float depth_pixel[2] = {(float)dx, (float)dy};
//             float depth_point[3];

//             rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth_in_meters);

//             /* store a vertex at the 3D location of this depth pixel */
//             pointcloud_[dy*depth_intrin.width*3 + dx*3 + 0] = depth_point[0];
//             pointcloud_[dy*depth_intrin.width*3 + dx*3 + 1] = depth_point[1];
//             pointcloud_[dy*depth_intrin.width*3 + dx*3 + 2] = depth_point[2];
//         }
//     }

//     npy_intp dims[3] = {depth_intrin.height, depth_intrin.width, 3};

//     return PyArray_SimpleNewFromData(
//         3,
//         dims,
//         NPY_FLOAT,
//         (void*) &pointcloud_
//         );
// }