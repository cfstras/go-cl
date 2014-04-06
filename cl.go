/*
This package provides wrappers around the OpenCL API.

To use this package, ensure that the headers and your favourite OpenCL
distribution is installed.
On Linux, the headers are usually available in a
package like "opencl-headers", while the implementation is provided by the
graphics driver distribution.
On Windows, your graphics driver distribution should have all the necessary stuff.

There are some basic tests provided as examples, see cl_test.go.


*/
package cl

//#cgo LDFLAGS: -lOpenCL
//#include "include.h"
import "C"
