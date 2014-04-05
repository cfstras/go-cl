package main

/*
#cgo LDFLAGS: -lOpenCL

#ifdef __APPLE__
#  include "OpenCL/opencl.h"
#else
#  include "CL/cl.h"
#endif

*/
import "C"

import (
	"fmt"
	"unsafe"
)

func main() {
	// get platform count
	var platformCount C.cl_uint
	C.clGetPlatformIDs(0, nil, &platformCount)
	fmt.Println("OpenCL platforms:", platformCount)
	// get platform IDs
	platformIDs := make([]C.cl_platform_id, platformCount)
	C.clGetPlatformIDs(platformCount, &platformIDs[0], nil)

	// get device count
	var deviceIdCount C.cl_uint
	C.clGetDeviceIDs(platformIDs[0], C.CL_DEVICE_TYPE_ALL, 0, nil, &deviceIdCount)
	fmt.Println("Devices:", deviceIdCount)
	// get device IDs
	deviceIDs := make([]C.cl_device_id, deviceIdCount)
	C.clGetDeviceIDs(platformIDs[0], C.CL_DEVICE_TYPE_ALL, deviceIdCount, &deviceIDs[0], nil)

	// make context
	contextProps := []C.cl_context_properties{
		C.CL_CONTEXT_PLATFORM,
		*(*C.cl_context_properties)(unsafe.Pointer(&platformIDs[0])),
		0,
	}

	var clErr C.cl_int
	C.clCreateContext(&contextProps[0], 1, &deviceIDs[0], nil, nil, &clErr)

}
