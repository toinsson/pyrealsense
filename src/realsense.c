/* License: Apache 2.0. See LICENSE file in root directory.
   Copyright(c) 2016 Antoine Loriette */


// Python Module includes
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include "rs.h"
#include "rsutil.h"

// global variables
void check_error(void);
void apply_ivcam_preset(rs_device * device, int preset);

rs_error * e = 0;
void check_error()
{
    if(e)
    {
        printf("rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e), rs_get_failed_args(e));
        printf("    %s\n", rs_get_error_message(e));
        exit(EXIT_FAILURE);
    }
}


/*
TODO: 
- rescale depth from get_depth - this is more or less a 1/8 or >>3 operation
- wrap the extrinsic and intrinsic structures
- memory statically allocated with only 1 buffer per stream
*/

// global variables
rs_context * ctx = NULL;
rs_device * dev = NULL;

// global camera parameters
rs_intrinsics depth_intrin, color_intrin;
rs_extrinsics depth_to_color;
float scale;

static PyObject *create_context(PyObject *self, PyObject *args, PyObject *keywds)
{
    // parse arguments
    int c_width = 640;
    int c_height = 480;
    int c_fps = 60;

    int d_width = 640;
    int d_height = 480;
    int d_fps = 60;

    int depth_control_preset = 0;
    int ivcam_preset = 9;  // optimised gesture recognition

    static char *kwlist[] = {"c_height", "c_width", "c_fps",
                             "d_height", "d_width", "d_fps",
                             "depth_control_preset", "ivcam_preset",
                             NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|iiiiiiii", kwlist,
                                     &c_height, &c_width, &c_fps,
                                     &d_height, &d_width, &d_fps,
                                     &depth_control_preset, &ivcam_preset
                                     ))
        return NULL;

    printf("c_height %d, c_width %d c_fps %d\n", c_height, c_width, c_fps);
    printf("d_height %d, d_width %d d_fps %d\n", d_height, d_width, d_fps);
    printf("depth_control_preset %d ivcam_preset %d \n", depth_control_preset, ivcam_preset);

    /* Create a context object. This object owns the handles to all connected realsense devices. */
    ctx = rs_create_context(RS_API_VERSION, &e);
    check_error();
    // printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    if(rs_get_device_count(ctx, &e) == 0)
    {
        check_error();
        Py_INCREF(Py_None);
        return Py_None;
    }

    /* Create a device object. */
    dev = rs_get_device(ctx, 0, &e);
    check_error();
    printf("\nUsing device 0, an %s\n", rs_get_device_name(dev, &e));
    check_error();
    printf("    Serial number: %s\n", rs_get_device_serial(dev, &e));
    check_error();
    printf("    Firmware version: %s\n", rs_get_device_firmware_version(dev, &e));
    check_error();

    // try out different options - SR300 are in preset 0 to 9
    // rs_apply_depth_control_preset(dev, depth_control_preset);
    // check_error();
    apply_ivcam_preset(dev, ivcam_preset);
    check_error();

    /* Configure all streams to run at VGA resolution at 60 frames per second */
    rs_enable_stream(dev, RS_STREAM_DEPTH, d_width, d_height, RS_FORMAT_Z16, d_fps, &e);
    check_error();
    rs_enable_stream(dev, RS_STREAM_COLOR, c_width, c_height, RS_FORMAT_RGB8, c_fps, &e);
    check_error();
    rs_enable_stream(dev, RS_STREAM_INFRARED, d_width, d_height, RS_FORMAT_Y8, c_fps, &e);
    check_error();

    // rs_enable_stream(dev, RS_STREAM_INFRARED2, d_width, d_height, RS_FORMAT_Y8, c_fps, NULL); /* Pass NULL to ignore errors */

    rs_start_device(dev, &e);
    check_error();

    /* Retrieve camera parameters for mapping between depth and color */
    // store locally at startup - this is static over porgram exec
    rs_get_stream_intrinsics(dev, RS_STREAM_DEPTH, &depth_intrin, &e);
    check_error();
    rs_get_device_extrinsics(dev, RS_STREAM_DEPTH, RS_STREAM_COLOR, &depth_to_color, &e);
    check_error();
    rs_get_stream_intrinsics(dev, RS_STREAM_COLOR, &color_intrin, &e);
    check_error();

    scale = rs_get_device_depth_scale(dev, &e);
    check_error();

    return Py_None;
}

// TODO for changing option on the fly
// static PyObject *set_device_option(PyObject *self, PyObject *args)
// {
//     // const double arr_values[15] = 
//     //     {1,     1, 100,  179,  179,   2,  16,  -1, 8000, 450,  1,  1,  7,  1, -1}

//     if (!PyArg_ParseTuple(args, "i", NULL, &depth_p))
//         return NULL;
// }


static PyObject *delete_context(PyObject *self, PyObject *args)
{
    rs_stop_device(dev, &e);
    // printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    rs_delete_context(ctx, &e);

    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject *get_colour(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();

    npy_intp dims[3] = {color_intrin.height, color_intrin.width, 3};

    return PyArray_SimpleNewFromData(
        3,
        dims,
        NPY_UINT8,
        (void*) rs_get_frame_data(dev, RS_STREAM_COLOR, &e)
        );
}


static PyObject *get_depth(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();

    npy_intp dims[2] = {depth_intrin.height, depth_intrin.width};

    return PyArray_SimpleNewFromData(
        2,
        dims,
        NPY_UINT16,
        (void*) rs_get_frame_data(dev, RS_STREAM_DEPTH, &e)
        );
}

static PyObject *get_depth_scale(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();
    float scale = rs_get_device_depth_scale(dev, &e);
    return PyFloat_FromDouble(scale);
}


static PyObject *get_ir(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();

    npy_intp dims[2] = {depth_intrin.height, depth_intrin.width};

    return PyArray_SimpleNewFromData(
        2,
        dims,
        NPY_UINT8,
        (void*) rs_get_frame_data(dev, RS_STREAM_INFRARED, &e)
        );
}


// local memory space for pointcloud - allocate max possible
float pointcloud[480*640*3];

static PyObject *get_pointcloud(PyObject *self, PyObject *args)
{
    rs_wait_for_frames(dev, &e);
    check_error();

    /* Retrieve image data */
    const uint16_t * depth_image = (const uint16_t *)rs_get_frame_data(dev, RS_STREAM_DEPTH, &e);
    check_error();


    memset(pointcloud, 0, sizeof(pointcloud));

    int dx, dy;
    for(dy=0; dy<depth_intrin.height; ++dy)
    {
        for(dx=0; dx<depth_intrin.width; ++dx)
        {
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
            pointcloud[dy*depth_intrin.width*3 + dx*3 + 0] = depth_point[0];
            pointcloud[dy*depth_intrin.width*3 + dx*3 + 1] = depth_point[1];
            pointcloud[dy*depth_intrin.width*3 + dx*3 + 2] = depth_point[2];
        }
    }

    npy_intp dims[3] = {depth_intrin.height, depth_intrin.width, 3};

    return PyArray_SimpleNewFromData(
        3,
        dims,
        NPY_FLOAT,
        (void*) &pointcloud
        );
}


// local memory space for the uv map
uint16_t uvmap[480*640*2];

static PyObject *get_uvmap(PyObject *self, PyObject *args)
{
    PyArrayObject *depth_p = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &depth_p))
        return NULL;

    // check we correct object - dims, type
    if (!(depth_p->nd == 2)) return NULL;

    memset(uvmap, 0, sizeof(uvmap));

    int dx, dy;
    for(dy=0; dy<depth_intrin.height; ++dy)
    {
        for(dx=0; dx<depth_intrin.width; ++dx)
        {
            /* Retrieve the 16-bit depth value and map it into a depth in meters */
            uint16_t depth_value = ((uint16_t*) depth_p->data)[dy * depth_intrin.width + dx];
            float depth_in_meters = depth_value * scale;

            /* Skip over pixels with a depth value of zero, which is used to indicate no data */
            if(depth_value == 0) continue;

            /* Map from pixel coordinates in the depth image to pixel coordinates in the color image */
            float depth_pixel[2] = {(float)dx, (float)dy};
            float depth_point[3], color_point[3], color_pixel[2];
            rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth_in_meters);
            rs_transform_point_to_point(color_point, &depth_to_color, depth_point);
            rs_project_point_to_pixel(color_pixel, &color_intrin, color_point);

            /* Use the color from the nearest color pixel, or pure white if this point falls outside the color image */
            const uint16_t cx = (uint16_t)roundf(color_pixel[0]);
            const uint16_t cy = (uint16_t)roundf(color_pixel[1]);

            if(cx < 0 || cy < 0 || cx >= color_intrin.width || cy >= color_intrin.height)
            {
                uvmap[dy*depth_intrin.width*2 + dx*2 + 0] = 0;
                uvmap[dy*depth_intrin.width*2 + dx*2 + 1] = 0;
            }
            else
            {
                uvmap[dy*depth_intrin.width*2 + dx*2 + 0] = cy;
                uvmap[dy*depth_intrin.width*2 + dx*2 + 1] = cx;
            }
        }
    }

    npy_intp dims[3] = {depth_intrin.height, depth_intrin.width, 2};

    return PyArray_SimpleNewFromData(
        3,
        dims,
        NPY_UINT16,
        (void*) &uvmap
        );
}


// local memory space for pointcloud - allocate max possible
float pointcloud_[480*640*3];

static PyObject *pointcloud_from_depth(PyObject *self, PyObject *args)
{
    PyArrayObject *depth_p = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &depth_p))
        return NULL;

    memset(pointcloud_, 0, sizeof(pointcloud_));

    int dx, dy;
    for(dy=0; dy<depth_intrin.height; ++dy)
    {
        for(dx=0; dx<depth_intrin.width; ++dx)
        {
            /* Retrieve the 16-bit depth value and map it into a depth in meters */
            uint16_t depth_value = ((uint16_t*) depth_p->data)[dy * depth_intrin.width + dx];
            float depth_in_meters = depth_value * scale;

            /* Skip over pixels with a depth value of zero, which is used to indicate no data */
            if(depth_value == 0) continue;

            /* Map from pixel coordinates in the depth image to pixel coordinates in the color image */
            float depth_pixel[2] = {(float)dx, (float)dy};
            float depth_point[3];

            rs_deproject_pixel_to_point(depth_point, &depth_intrin, depth_pixel, depth_in_meters);

            /* store a vertex at the 3D location of this depth pixel */
            pointcloud_[dy*depth_intrin.width*3 + dx*3 + 0] = depth_point[0];
            pointcloud_[dy*depth_intrin.width*3 + dx*3 + 1] = depth_point[1];
            pointcloud_[dy*depth_intrin.width*3 + dx*3 + 2] = depth_point[2];
        }
    }

    npy_intp dims[3] = {depth_intrin.height, depth_intrin.width, 3};

    return PyArray_SimpleNewFromData(
        3,
        dims,
        NPY_FLOAT,
        (void*) &pointcloud_
        );
}


// local memory map for the ouput pixel
uint16_t depth_pixel_round[2];

static PyObject *project_point_to_pixel(PyObject *self, PyObject *args)
{
    PyArrayObject *point_p = NULL;

    // get the depth and colour image
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &point_p))
        return NULL;

    // check we correct object - dims, type
    // use assert
    if (!(point_p->nd == 1)) return NULL;

    float depth_point[3];


    depth_point[0] = ((float*)point_p->data)[0];
    depth_point[1] = ((float*)point_p->data)[1];
    depth_point[2] = ((float*)point_p->data)[2];

    printf("%f %f %f \n", depth_point[0], depth_point[1], depth_point[2]);


    float depth_pixel[2];
    rs_project_point_to_pixel(depth_pixel, &depth_intrin, depth_point);

    const int cx = (int)roundf(depth_pixel[0]), cy = (int)roundf(depth_pixel[1]);
    if(cx < 0 || cy < 0 || cx >= depth_intrin.width || cy >= depth_intrin.height)
    {
        depth_pixel_round[0] = 0;
        depth_pixel_round[1] = 0;
    }
    else
    {
        depth_pixel_round[0] = cy;
        depth_pixel_round[1] = cx;
    }

    npy_intp dims[1] = {2};

    return PyArray_SimpleNewFromData(
        1,
        dims,
        NPY_UINT16,
        (void*) &depth_pixel_round
        );
}

static PyObject *get_rs_extrinsics(PyObject *self, PyObject *args)
{
    // typedef struct rs_extrinsics
    // {
    //     float rotation[9];    // column-major 3x3 rotation matrix 
    //     float translation[3]; // 3 element translation vector, in meters 
    // } rs_extrinsics;

    printf("depth_to_color rotation[9] translation[3]\n");
    printf("%f %f %f %f %f %f %f %f %f\n",
    depth_to_color.rotation[0],
    depth_to_color.rotation[1],
    depth_to_color.rotation[2],
    depth_to_color.rotation[3],
    depth_to_color.rotation[4],
    depth_to_color.rotation[5],
    depth_to_color.rotation[6],
    depth_to_color.rotation[7],
    depth_to_color.rotation[8]
    );

    printf("%f %f %f\n",
    depth_to_color.translation[0],
    depth_to_color.translation[1],
    depth_to_color.translation[2]
    );

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *get_rs_intrinsics(PyObject *self, PyObject *args)
{
    // typedef struct rs_intrinsics
    // {
    // int           width;     // width of the image in pixels 
    // int           height;    // height of the image in pixels 
    // float         ppx;       // horizontal coordinate of the principal point of the image, as a pixel offset from the left edge 
    // float         ppy;       // vertical coordinate of the principal point of the image, as a pixel offset from the top edge 
    // float         fx;        // focal length of the image plane, as a multiple of pixel width 
    // float         fy;        // focal length of the image plane, as a multiple of pixel height 
    // rs_distortion model;     // distortion model of the image 
    // float         coeffs[5]; // distortion coefficients 
    // } rs_intrinsics;

    printf("color width height ppx ppy fx fy coeffs[5]\n");
    printf("%d %d %f %f %f %f %f %f %f %f %f\n",
                                color_intrin.width,
                                color_intrin.height,
                                color_intrin.ppx,
                                color_intrin.ppy,
                                color_intrin.fx,
                                color_intrin.fy,
                                color_intrin.coeffs[0],
                                color_intrin.coeffs[1],
                                color_intrin.coeffs[2],
                                color_intrin.coeffs[3],
                                color_intrin.coeffs[4]
                                );

    printf("depth width height ppx ppy fx fy coeffs[5]\n");
    printf("%d %d %f %f %f %f %f %f %f %f %f\n",
                            depth_intrin.width,
                            depth_intrin.height,
                            depth_intrin.ppx,
                            depth_intrin.ppy,
                            depth_intrin.fx,
                            depth_intrin.fy,
                            depth_intrin.coeffs[0],
                            depth_intrin.coeffs[1],
                            depth_intrin.coeffs[2],
                            depth_intrin.coeffs[3],
                            depth_intrin.coeffs[4]
                            );

    Py_INCREF(Py_None);
    return Py_None;
}


/* Provide access to several recommend sets of option presets for ivcam */
void apply_ivcam_preset(rs_device * device, int preset)
{
    const rs_option arr_options[15] = {
        RS_OPTION_SR300_AUTO_RANGE_ENABLE_MOTION_VERSUS_RANGE,  //00
        RS_OPTION_SR300_AUTO_RANGE_ENABLE_LASER,                //01
        RS_OPTION_SR300_AUTO_RANGE_MIN_MOTION_VERSUS_RANGE,     //02
        RS_OPTION_SR300_AUTO_RANGE_MAX_MOTION_VERSUS_RANGE,     //03
        RS_OPTION_SR300_AUTO_RANGE_START_MOTION_VERSUS_RANGE,   //04
        RS_OPTION_SR300_AUTO_RANGE_MIN_LASER,                   //05
        RS_OPTION_SR300_AUTO_RANGE_MAX_LASER,                   //06
        RS_OPTION_SR300_AUTO_RANGE_START_LASER,                 //07
        RS_OPTION_SR300_AUTO_RANGE_UPPER_THRESHOLD,             //08
        RS_OPTION_SR300_AUTO_RANGE_LOWER_THRESHOLD,             //09
        RS_OPTION_F200_LASER_POWER,                             //10
        RS_OPTION_F200_ACCURACY,                                //11
        RS_OPTION_F200_FILTER_OPTION,                           //12
        RS_OPTION_F200_CONFIDENCE_THRESHOLD,                    //13
        RS_OPTION_F200_MOTION_RANGE                             //14
    };

    const double arr_values[][15] = {
      //00     01   02    03    04   05   06   07    08   09  10  11  12  13  14
         /*00 Common                 */
        {1,     1, 180,  605,  303,   2,  16,  -1, 1250, 650,  1,  1,  5,  1, -1},
        /* 01 ShortRange             */
        {1,     1, 180,  303,  180,   2,  16,  -1, 1000, 450,  1,  1,  5,  1, -1},
        /* 02 LongRange              */
        {1,     0, 303,  605,  303,  -1,  -1,  -1, 1250, 975,  1,  1,  7,  0, -1},
        /* 03 BackgroundSegmentation */
        {0,     0,  -1,   -1,   -1,  -1,  -1,  -1,   -1,  -1, 16,  1,  6,  0, 22},
        /* 04 GestureRecognition     */
        {1,     1, 100,  179,  100,   2,  16,  -1, 1000, 450,  1,  1,  6,  3, -1},
        /* 05 ObjectScanning         */
        {0,     1,  -1,   -1,   -1,   2,  16,  16, 1000, 450,  1,  1,  3,  1,  9},
        /* 06 FaceMW                 */
        {0,     0,  -1,   -1,   -1,  -1,  -1,  -1,   -1,  -1, 16,  1,  5,  1, 22},
        /* 07 FaceLogin              */
        {2,     0,  40, 1600,  800,  -1,  -1,  -1,   -1,  -1,  1, -1, -1, -1, -1},
        /* 08 GRCursorMode           */
        {1,     1, 100,  179,  179,   2,  16,  -1, 1000, 450,  1,  1,  6,  1, -1},
        /* 09 custom - gesture long range */
        {1,     1, 100,  179,  100,   2,  16,  -1, 2500, 450,  1,  1,  6,  7, -1}
    };

    rs_error * e = 0;

    for (int i=0; i<15; i++)
    {
        printf("%f ", arr_values[preset][i]);
    }
    printf("\n");


    if(arr_values[preset][14] != -1)
    {
        rs_set_device_options(device, arr_options, 15, arr_values[preset], &e);
        if(e)
        {
            printf("rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e),
                                                                 rs_get_failed_args(e));
            printf("    %s\n", rs_get_error_message(e));
        }
    }

    if(arr_values[preset][13] != -1)
    {
        rs_set_device_options(device, arr_options, 14, arr_values[preset], &e);
        if(e)
        {
            printf("rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e),
                                                                 rs_get_failed_args(e));
            printf("    %s\n", rs_get_error_message(e));
        }
    }

    else
    {
        rs_set_device_options(device, arr_options, 11, arr_values[preset], &e);
        if(e)
        {
            printf("rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e),
                                                                 rs_get_failed_args(e));
            printf("    %s\n", rs_get_error_message(e));
        }
    }
}


static PyMethodDef RealSenseMethods[] = {
    // GET MAPS
    {"get_colour",  get_colour, METH_VARARGS, "Get colour map"},
    {"get_depth",  get_depth, METH_VARARGS, "Get depth map"},
    {"get_ir",  get_ir, METH_VARARGS, "Get ir map"},
    {"get_depth_scale",  get_depth_scale, METH_VARARGS, "Get Depth Scale"},
    {"get_pointcloud",  get_pointcloud, METH_VARARGS, "Get point cloud"},

    {"get_uvmap",  get_uvmap, METH_VARARGS, "Get UV map"},
    {"pointcloud_from_depth",  pointcloud_from_depth, METH_VARARGS, "Get the pointcloud from depth."},
    {"project_point_to_pixel",  project_point_to_pixel, METH_VARARGS, "Project point to pixel."},

    {"get_rs_intrinsics",  get_rs_intrinsics, METH_VARARGS, "Get intrinsic parameters."},
    {"get_rs_extrinsics",  get_rs_extrinsics, METH_VARARGS, "Get extrinsic parameters."},


    // // CREATE MODULE
    {"start", (PyCFunction)create_context, METH_VARARGS | METH_KEYWORDS, "Start RealSense"},
    {"close", delete_context, METH_VARARGS, "Close DepthSense"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

void init_default_camera_parameters(void);
void init_default_camera_parameters(void)
{
    scale = 0.00012498664727900177;

    // save static values for calling projection functions when camera not present
    depth_to_color.rotation[0] =  0.999998;
    depth_to_color.rotation[1] = -0.001382;
    depth_to_color.rotation[2] =  0.001113;
    depth_to_color.rotation[3] =  0.001388;
    depth_to_color.rotation[4] =  0.999982;
    depth_to_color.rotation[5] = -0.005869;
    depth_to_color.rotation[6] = -0.001104;
    depth_to_color.rotation[7] =  0.005871;
    depth_to_color.rotation[8] =  0.999982;

    depth_to_color.translation[0] =  0.025700;
    depth_to_color.translation[1] = -0.000733;
    depth_to_color.translation[2] =  0.003885;

    // save static values for calling projection functions when camera not present
    color_intrin.width      = 640;
    color_intrin.height     = 480;
    color_intrin.ppx        = 310.672333;
    color_intrin.ppy        = 249.916473;
    color_intrin.fx         = 613.874634;
    color_intrin.fy         = 613.874695;
    color_intrin.coeffs[0]  = 0.0;
    color_intrin.coeffs[1]  = 0.0;
    color_intrin.coeffs[2]  = 0.0;
    color_intrin.coeffs[3]  = 0.0;
    color_intrin.coeffs[4]  = 0.0;

    depth_intrin.width      = 640;
    depth_intrin.height     = 480;
    depth_intrin.ppx        = 314.796814;
    depth_intrin.ppy        = 245.890991;
    depth_intrin.fx         = 475.529053;
    depth_intrin.fy         = 475.528931;
    depth_intrin.coeffs[0]  = 0.144294;
    depth_intrin.coeffs[1]  = 0.054764;
    depth_intrin.coeffs[2]  = 0.004520;
    depth_intrin.coeffs[3]  = 0.002106;
    depth_intrin.coeffs[4]  = 0.107831;
}



PyMODINIT_FUNC initpyrealsense(void)
{
    (void) Py_InitModule3("pyrealsense", RealSenseMethods, "Simple C extension to librealsense.");
    import_array();

    // init default values
    init_default_camera_parameters();
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
