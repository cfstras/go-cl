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

type Context struct {
	cl C.cl_context
}

type Buffer C.cl_mem

func CreateContext() *Context {
	// get platform count
	var platformCount C.cl_uint
	C.clGetPlatformIDs(0, nil, &platformCount)
	fmt.Println("OpenCL platforms:", platformCount)
	// get platform IDs
	platformIDs := make([]C.cl_platform_id, platformCount)
	C.clGetPlatformIDs(platformCount, &platformIDs[0], nil)

	// get device count
	var deviceIdCount C.cl_uint
	C.clGetDeviceIDs(platformIDs[0], C.CL_DEVICE_TYPE_ALL, 0, nil,
		&deviceIdCount)
	fmt.Println("Devices:", deviceIdCount)
	// get device IDs
	deviceIDs := make([]C.cl_device_id, deviceIdCount)
	C.clGetDeviceIDs(platformIDs[0], C.CL_DEVICE_TYPE_ALL, deviceIdCount,
		&deviceIDs[0], nil)

	// make context
	contextProps := []C.cl_context_properties{
		C.CL_CONTEXT_PLATFORM,
		*(*C.cl_context_properties)(unsafe.Pointer(&platformIDs[0])),
		0,
	}

	var clErr C.cl_int
	context := C.clCreateContext(&contextProps[0], 1, &deviceIDs[0], nil, nil,
		&clErr)
	if clErr != C.CL_SUCCESS {
		fmt.Println("CL error", clErr, "!")
		return nil
	}
	return &Context{context}
}

func (c *Context) CreateBuffer(data []float32) Buffer {
	var clErr C.cl_int
	var buffer C.cl_mem
	buffer = C.clCreateBuffer(c.cl, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(int(unsafe.Sizeof(data[0]))*len(data)), unsafe.Pointer(&data[0]),
		&clErr)

	if clErr != C.CL_SUCCESS {
		fmt.Println("CL error", clErr, "!")
		return nil
	}
	return Buffer(buffer)
}

func main() {
	ctx := CreateContext()

	x := []float32{1, 2, 3, 4}
	buffer := ctx.CreateBuffer(x)

	fmt.Println(buffer)
}
