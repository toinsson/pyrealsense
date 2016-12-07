#include "rs.h"
#include "rsutil.h"
#include "assert.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>


void _apply_depth_control_preset(rs_device * device, int preset)
{
    rs_apply_depth_control_preset(device, preset);
}

void _apply_ivcam_preset(rs_device * device, rs_ivcam_preset preset)
{
    rs_apply_ivcam_preset(device, preset);
}
