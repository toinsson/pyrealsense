// fc.h: numpy arrays from cython , double*

#include "rs.h"

int fc( int N, const double a[], const double b[], double z[] );

void _apply_depth_control_preset(rs_device * device, int preset);
void _apply_ivcam_preset(rs_device * device, rs_ivcam_preset preset);
// const void * _project_point_to_pixel(const void * point, const rs_intrinsics * intrin);
void _project_point_to_pixel(float pixel[], const rs_intrinsics * intrin, const float point[]);
void _deproject_pixel_to_point(float point[], const struct rs_intrinsics * intrin, const float pixel[], float depth);
