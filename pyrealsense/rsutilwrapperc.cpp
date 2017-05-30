#include <stdint.h>

#include "rs.h"
#include "rsutil.h"


void _apply_depth_control_preset(rs_device * device, int preset)
{
    rs_apply_depth_control_preset(device, preset);
}


void _apply_ivcam_preset(rs_device * device, rs_ivcam_preset preset)
{
    rs_apply_ivcam_preset(device, preset);
}


void _project_point_to_pixel(float pixel[],
                             const struct rs_intrinsics * intrin,
                             const float point[])
{
    rs_project_point_to_pixel(pixel, intrin, point);
}


void _deproject_pixel_to_point(float point[],
                               const struct rs_intrinsics * intrin,
                               const float pixel[],
                               float depth)
{
    rs_deproject_pixel_to_point(point, intrin, pixel, depth);
}


void _deproject_depth(float pointcloud[],
                      const struct rs_intrinsics * intrin,
                      const uint16_t depth_image[],
                      const float depth_scale)
{
    int dx, dy;
    for(dy=0; dy<intrin->height; ++dy)
    {
        for(dx=0; dx<intrin->width; ++dx)
        {
            /* Retrieve the 16-bit depth value and map it into a depth in meters */
            uint16_t depth_value = ((uint16_t*)depth_image)[dy * intrin->width + dx];
            float depth_in_meters = depth_value * depth_scale;
            /* Skip over pixels with a depth value of zero, which is used to indicate no data */
            if(depth_value == 0) continue;
            /* Map from pixel coordinates in the depth image to pixel coordinates in the color image */
            float depth_pixel[2] = {(float)dx, (float)dy};
            float depth_point[3];
            rs_deproject_pixel_to_point(depth_point, intrin, depth_pixel, depth_in_meters);
            /* store a vertex at the 3D location of this depth pixel */
            pointcloud[dy*intrin->width*3 + dx*3 + 0] = depth_point[0];
            pointcloud[dy*intrin->width*3 + dx*3 + 1] = depth_point[1];
            pointcloud[dy*intrin->width*3 + dx*3 + 2] = depth_point[2];
        }
    }
}

