#include "rs.h"
#include "rsutil.h"
#include "assert.h"

#include <stdio.h>
#include <string.h>

// #ifdef WIN_PYTHON_2
// #include "stdint.h"
// #else
#include <stdint.h>
// #endif

#include <Python.h>

void _apply_depth_control_preset(rs_device * device, int preset)
{
    rs_apply_depth_control_preset(device, preset);
}


void _apply_ivcam_preset(rs_device * device, rs_ivcam_preset preset)
{
    rs_apply_ivcam_preset(device, preset);
}


float pixel[2];
const void * project_point_to_pixel(const void * point, const rs_intrinsics * intrin)
{
    rs_project_point_to_pixel(pixel, intrin, (const float *)point);
    return pixel;
}


float point[3];
const void * deproject_pixel_to_point(const void * pixel, const float depth, const rs_intrinsics * intrin)
{
    rs_deproject_pixel_to_point(point, intrin, (const float *)pixel, depth);
    return point;
}


// local memory space for pointcloud - max size
float pointcloud[480*640*3];

const void * deproject_depth(const void * depth_image,
                            const rs_intrinsics * depth_intrin,
                            const float * depth_scale)
{
    memset(pointcloud, 0, sizeof(pointcloud));
    int dx, dy;
    for(dy=0; dy<depth_intrin->height; ++dy)
    {
        for(dx=0; dx<depth_intrin->width; ++dx)
        {
            /* Retrieve the 16-bit depth value and map it into a depth in meters */
            uint16_t depth_value = ((uint16_t*)depth_image)[dy * depth_intrin->width + dx];
            float depth_in_meters = depth_value * depth_scale[0];
            /* Skip over pixels with a depth value of zero, which is used to indicate no data */
            if(depth_value == 0) continue;
            /* Map from pixel coordinates in the depth image to pixel coordinates in the color image */
            float depth_pixel[2] = {(float)dx, (float)dy};
            float depth_point[3];
            rs_deproject_pixel_to_point(depth_point, depth_intrin, depth_pixel, depth_in_meters);
            /* store a vertex at the 3D location of this depth pixel */
            pointcloud[dy*depth_intrin->width*3 + dx*3 + 0] = depth_point[0];
            pointcloud[dy*depth_intrin->width*3 + dx*3 + 1] = depth_point[1];
            pointcloud[dy*depth_intrin->width*3 + dx*3 + 2] = depth_point[2];
        }
    }
    return pointcloud;
}

// not used for anything since this isn't a real Python extension,
// just keeps compiler happy on Windows
PyMODINIT_FUNC initrsutilwrapper(void) 
{

}
