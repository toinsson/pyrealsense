

// Python Module includes
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "rs.h"
#include "rsutil.h"

// global variables
rs_context * ctx = NULL;
rs_error * e = 0;
rs_device * dev = NULL;

void check_error()
{
    if(e)
    {
        printf("rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e), rs_get_failed_args(e));
        printf("    %s\n", rs_get_error_message(e));
        exit(EXIT_FAILURE);
    }
}


static PyObject *createContext(PyObject *self, PyObject *args)
{
    /* Create a context object. This object owns the handles to all connected realsense devices. */
    ctx = rs_create_context(RS_API_VERSION, &e);
    check_error();
    // printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    // check_error();
    if(rs_get_device_count(ctx, &e) == 0) return Py_None;

    /* This tutorial will access only a single device, but it is trivial to extend to multiple devices */
    dev = rs_get_device(ctx, 0, &e);
    check_error();
    // printf("\nUsing device 0, an %s\n", rs_get_device_name(dev, &e));
    // check_error();
    // printf("    Serial number: %s\n", rs_get_device_serial(dev, &e));
    // check_error();
    // printf("    Firmware version: %s\n", rs_get_device_firmware_version(dev, &e));
    // check_error();

    /* Configure all streams to run at VGA resolution at 60 frames per second */
    rs_enable_stream(dev, RS_STREAM_DEPTH, 640, 480, RS_FORMAT_Z16, 60, &e);
    check_error();
    rs_enable_stream(dev, RS_STREAM_COLOR, 640, 480, RS_FORMAT_RGB8, 60, &e);
    check_error();
    rs_enable_stream(dev, RS_STREAM_INFRARED, 640, 480, RS_FORMAT_Y8, 60, &e);
    check_error();
    rs_enable_stream(dev, RS_STREAM_INFRARED2, 640, 480, RS_FORMAT_Y8, 60, NULL); /* Pass NULL to ignore errors */
    rs_start_device(dev, &e);
    check_error();

    return Py_None;

}


static PyObject *deleteContext(PyObject *self, PyObject *args)
{
    rs_stop_device(dev, &e);
    // printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    rs_delete_context(ctx, &e);

    return Py_None;
}


static PyObject *getDepthScale(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();
    float scale = rs_get_device_depth_scale(dev, &e);
    return PyFloat_FromDouble(scale);
}


static PyObject *getColour(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();

    npy_intp dims[3] = {480, 640, 3};

    return PyArray_SimpleNewFromData(
        3,
        dims,
        NPY_UINT8,
        (void*) rs_get_frame_data(dev, RS_STREAM_COLOR, &e)
        );
}


static PyObject *getDepth(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();

    npy_intp dims[2] = {480, 640};

    return PyArray_SimpleNewFromData(
        2,
        dims,
        NPY_UINT16,
        (void*) rs_get_frame_data(dev, RS_STREAM_DEPTH, &e)
        );
}

float pointcloud[480*640*3];


static PyObject *getPointCloud(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();

    /* Retrieve image data */
    const uint16_t * depth_image = (const uint16_t *)rs_get_frame_data(dev, RS_STREAM_DEPTH, &e);
    check_error();
    const uint8_t * color_image = (const uint8_t *)rs_get_frame_data(dev, RS_STREAM_COLOR, &e);
    check_error();

    /* Retrieve camera parameters for mapping between depth and color */
    rs_intrinsics depth_intrin, color_intrin;
    rs_extrinsics depth_to_color;
    rs_get_stream_intrinsics(dev, RS_STREAM_DEPTH, &depth_intrin, &e);
    check_error();
    rs_get_device_extrinsics(dev, RS_STREAM_DEPTH, RS_STREAM_COLOR, &depth_to_color, &e);
    check_error();
    rs_get_stream_intrinsics(dev, RS_STREAM_COLOR, &color_intrin, &e);
    check_error();
    float scale = rs_get_device_depth_scale(dev, &e);
    check_error();

    int dx, dy;
    for(dy=0; dy<depth_intrin.height; ++dy)
    {
        for(dx=0; dx<depth_intrin.width; ++dx)
        {
            pointcloud[dy*640*3 + 3*dx + 0] = 0;
            pointcloud[dy*640*3 + 3*dx + 1] = 0;
            pointcloud[dy*640*3 + 3*dx + 2] = 0;

            /* Retrieve the 16-bit depth value and map it into a depth in meters */
            uint16_t depth_value = depth_image[dy * depth_intrin.width + dx];
            float depth_in_meters = depth_value * scale;

            /* Skip over pixels with a depth value of zero, which is used to indicate no data */
            if(depth_value == 0) continue;

            /* Map from pixel coordinates in the depth image to pixel coordinates in the color image */
            float depth_pixel[2] = {(float)dx, (float)dy};
            float depth_point[3];

            rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth_in_meters);

            /* store a vertex at the 3D location of this depth pixel */
            // memset(pointcloud, 0, sizeof(pointcloud));
            pointcloud[dy*640*3 + 3*dx + 0] = depth_point[0];
            pointcloud[dy*640*3 + 3*dx + 1] = depth_point[1];
            pointcloud[dy*640*3 + 3*dx + 2] = depth_point[2];
        }
    }

    npy_intp dims[3] = {640, 480, 3};

    return PyArray_SimpleNewFromData(
        3,
        dims,
        NPY_FLOAT,
        (void*) &pointcloud
        );
}


static PyMethodDef RealSenseMethods[] = {
    // GET MAPS
    {"get_colour_map",  getColour, METH_VARARGS, "Get Colour Map"},
    {"get_depth_map",  getDepth, METH_VARARGS, "Get Depth Map"},
    {"get_point_cloud",  getPointCloud, METH_VARARGS, "Get Point Cloud"},


    {"get_depth_scale",  getDepthScale, METH_VARARGS, "Get Depth Scale"},

    // // CREATE MODULE
    {"start", createContext, METH_VARARGS, "Start RealSense"},
    {"close", deleteContext, METH_VARARGS, "Close DepthSense"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC initpyrealsense(void)
{
    (void) Py_InitModule("pyrealsense", RealSenseMethods);
    import_array();
}


int main(int argc, char* argv[])
{
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName((char *)"RealSense");

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initpyrealsense();

    return 0;
}
