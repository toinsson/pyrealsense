#include "rs.h"
#include "rsutil.h"
#include "assert.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

void print_array(float point_[3], float* point__)
{
    for (int i = 0; i < 3; i++)
    {
        printf("%f ", point_[i]);
        printf("%f ", point__[i]);
    }
}

// local memory space for pointcloud - allocate max possible
float pointcloud[480*640*3];

const void * get_pointcloud(const void * depth_image,
                            const rs_intrinsics * depth_intrin,
                            const float * depth_scale)
{

    // uint16_t* depth_data = (uint16_t*)depth_image;

    // printf("%f \n", *depth_scale);

    // printf("depth_intrin %d %d \n", depth_intrin->height, depth_intrin->width);

    // printf("depth_image\n");
    // for (int i=0; i< 10; i++){
    //     printf("%d ", depth_data[i]);
    // }
    // printf("\n");
    // for (int i=200; i< 240; i++){
    //     printf("%d ", depth_data[i]);
    // }
    // printf("\n");

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
    // npy_intp dims[3] = {depth_intrin.height, depth_intrin.width, 3};

    // return PyArray_SimpleNewFromData(
    //     3,
    //     dims,
    //     NPY_FLOAT,
    //     (void*) &pointcloud
    //     );
}