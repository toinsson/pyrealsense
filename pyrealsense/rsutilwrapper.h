// fc.h: numpy arrays from cython , double*

#include "rs.h"

int fc( int N, const double a[], const double b[], double z[] );

void _apply_depth_control_preset(rs_device * device, int preset);
void _apply_ivcam_preset(rs_device * device, rs_ivcam_preset preset);