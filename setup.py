import sys
from platform import system
from distutils.core import setup, Extension
import numpy

##
## compatibility checks
##
try: from exceptions import NotImplementedError
except ImportError: pass

if sys.version_info >= (3, 0):  # due to C extension wrapping
    raise NotImplementedError("Python 3.x is not supported.")

ostype = system()
if ostype != 'Linux' and ostype != 'Windows':
    raise NotImplementedError("Only Windows and Linux supported.")


##
## Platform dependant configuration
##
is_64bits = sys.maxsize > 2**32

## windows platform
# ## Windows 32bits
# if ostype == 'Windows' and is_64bits == False:
#     depthsensesdk_path = "C:\\Program Files (x86)\\SoftKinetic\\DepthSenseSDK\\"
#     additional_include = './inc'
#     compile_args = ['/EHsc']
# ## Windows 64bits
# elif ostype == 'Windows' and is_64bits == True:
#     depthsensesdk_path = "C:\\Program Files\\SoftKinetic\\DepthSenseSDK\\"
#     additional_include = './inc'
#     compile_args = ['/EHsc']


## Linux
if ostype == 'Linux':
    depthsensesdk_path = "/opt/softkinetic/DepthSenseSDK/"
    additional_include = './'
    compile_args = []
else:
    raise NotImplementedError("Only Windows and Linux supported.")

modname = 'pyrealsense'
libnames = ['realsense']
sourcefiles = ['src/realsense.cxx',]# 'src/initdepthsense.cxx']

module = Extension(modname,
    include_dirs = [numpy.get_include(),"/usr/local/include/librealsense"],
    libraries = libnames,
    library_dirs = ["/usr/local/lib"],
    # extra_compile_args = compile_args,
    sources = sourcefiles)

setup (name = 'pyrealsense',
        version = '1.0',
        # description = 'Python wrapper for the Senz3d camera under Linux.',
        # author = 'Antoine Loriette',
        # url = 'https://github.com/toinsson/pysenz3d-linux',
        # long_description = '''This wrapper provides the main functionality of the DS325, aka the
        # Creative Senz3d camera. It is based on the Softkinetic demo code and was kicked started from
        # the Github project of ...
        # ''',
        ext_modules = [module])
