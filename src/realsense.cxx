

// Python Module includes
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "rs.h"


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
    printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    check_error();
    if(rs_get_device_count(ctx, &e) == 0) return Py_None;

    /* This tutorial will access only a single device, but it is trivial to extend to multiple devices */
    dev = rs_get_device(ctx, 0, &e);
    check_error();
    printf("\nUsing device 0, an %s\n", rs_get_device_name(dev, &e));
    check_error();
    printf("    Serial number: %s\n", rs_get_device_serial(dev, &e));
    check_error();
    printf("    Firmware version: %s\n", rs_get_device_firmware_version(dev, &e));
    check_error();

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

    // /* Open a GLFW window to display our output */
    // glfwInit();
    // GLFWwindow * win = glfwCreateWindow(1280, 960, "librealsense tutorial #2", NULL, NULL);
    // glfwMakeContextCurrent(win);
    return Py_None;

}

static PyObject *deleteContext(PyObject *self, PyObject *args)
{
    rs_stop_device(dev, &e);
    printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    rs_delete_context(ctx, &e);
    ctx = NULL;
    dev = NULL;

    return Py_None;
}

static int32_t dW = 320;
static int32_t dH = 240;
static int32_t cW = 640;
static int32_t cH = 480;
static int dshmsz = dW*dH*sizeof(int16_t);
int16_t depthMapClone[320*240];



static PyObject *getDepth(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();

    printf("new frame\n");

    rs_get_device_depth_scale(dev, &e);
    printf("get device depth scale\n");

    rs_get_frame_data(dev, RS_STREAM_DEPTH, &e);
    printf("get frame rs_get_frame_data\n");

    // npy_intp dims[2] = {640, 480};
    // memcpy(depthMapClone, rs_get_frame_data(dev, RS_STREAM_DEPTH, &e), dshmsz);

    // printf("memcpy");

    // return PyArray_SimpleNewFromData(
    //     2,
    //     dims,
    //     NPY_UINT16,
    //     depthMapClone
    //     );

    return Py_None;
    // npy_intp dims[2] = {dH, dW};
    // memcpy(depthMapClone, depthFullMap, dshmsz);
    // return PyArray_SimpleNewFromData(2, dims, NPY_INT16, depthMapClone);
}

static PyMethodDef RealSenseMethods[] = {
    // GET MAPS
    {"get_depth_map",  getDepth, METH_VARARGS, "Get Depth Map"},
    // {"getColourMap",  getColour, METH_VARARGS, "Get Colour Map"},
    // {"getVertices",  getVertex, METH_VARARGS, "Get Vertex Map"},
    // {"getVerticesFP",  getVertexFP, METH_VARARGS, "Get Floating Point Vertex Map"},
    // {"getUVMap",  getUV, METH_VARARGS, "Get UV Map"},
    // {"getSyncMap",  getSync, METH_VARARGS, "Get Colour Overlay Map"},
    // {"getAcceleration",  getAccel, METH_VARARGS, "Get Acceleration"},
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
