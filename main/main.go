package main

/*
#cgo LDFLAGS: -lOpenCL

#include <stdlib.h>

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
	C_context C.cl_context
	Queue     C.cl_command_queue

	NumDevices C.cl_uint
	DeviceIDs  []C.cl_device_id

	Err C.cl_int
}

type Buffer C.cl_mem
type Kernel C.cl_kernel

func CreateContext() *Context {
	c := &Context{}

	// get platform count
	var platformCount C.cl_uint
	C.clGetPlatformIDs(0, nil, &platformCount)
	fmt.Println("OpenCL platforms:", platformCount)
	// get platform IDs
	platformIDs := make([]C.cl_platform_id, platformCount)
	C.clGetPlatformIDs(platformCount, &platformIDs[0], nil)

	// get device count
	C.clGetDeviceIDs(platformIDs[0], C.CL_DEVICE_TYPE_ALL, 0, nil,
		&c.NumDevices)
	fmt.Println("Devices:", c.NumDevices)
	// get device IDs
	c.DeviceIDs = make([]C.cl_device_id, c.NumDevices)
	C.clGetDeviceIDs(platformIDs[0], C.CL_DEVICE_TYPE_ALL, c.NumDevices,
		&c.DeviceIDs[0], nil)

	// make context
	contextProps := []C.cl_context_properties{
		C.CL_CONTEXT_PLATFORM,
		*(*C.cl_context_properties)(unsafe.Pointer(&platformIDs[0])),
		0,
	}

	c.C_context = C.clCreateContext(&contextProps[0], 1, &c.DeviceIDs[0], nil, nil,
		&c.Err)
	c.CheckErr()

	c.Queue = C.clCreateCommandQueue(c.C_context, c.DeviceIDs[0], 0, &c.Err)
	c.CheckErr()

	return c
}

func (c *Context) CreateBuffer(data []float32) Buffer {
	var buffer C.cl_mem
	buffer = C.clCreateBuffer(c.C_context, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(int(unsafe.Sizeof(data[0]))*len(data)), unsafe.Pointer(&data[0]),
		&c.Err)
	c.CheckErr()
	return Buffer(buffer)
}

func (c *Context) CompileProgram(code, mainFunction string) Kernel {
	str := C.CString(sourceCode)
	defer C.free(unsafe.Pointer(str))
	prog := C.clCreateProgramWithSource(c.C_context, 1, &str, nil, &c.Err)
	c.CheckErr()

	C.clBuildProgram(prog, c.NumDevices, &c.DeviceIDs[0], nil, nil, nil)
	c.CheckErr()

	mainFunc := C.CString(mainFunction)
	defer C.free(unsafe.Pointer(mainFunc))
	kernel := C.clCreateKernel(prog, mainFunc, &c.Err)
	c.CheckErr()
	return Kernel(kernel)
}

func (c *Context) CheckErr() {
	if c.Err != C.CL_SUCCESS {
		fmt.Println("CL error", c.Err, "!")
		panic("CL error")
	}
}

const sourceCode = `
__kernel void SAXPY (__global float* x, __global float* y, float a)
{
    const int i = get_global_id (0);

    y [i] += a * x [i];
}`

func main() {
	ctx := CreateContext()

	x := []float32{1, 2, 3, 4}
	y := make([]float32, 4)
	buffer := ctx.CreateBuffer(x)
	dstBuffer := ctx.CreateBuffer(y)

	kernel := ctx.CompileProgram(sourceCode, "SAXPY")

	C.clSetKernelArg(kernel, 0, C.size_t(unsafe.Sizeof(buffer)),
		unsafe.Pointer(&buffer))
	C.clSetKernelArg(kernel, 1, C.size_t(unsafe.Sizeof(dstBuffer)),
		unsafe.Pointer(&dstBuffer))

	var mult float32 = 2.0
	C.clSetKernelArg(kernel, 2, C.size_t(unsafe.Sizeof(mult)),
		unsafe.Pointer(&mult))

	// start!
	workSize := []C.size_t{C.size_t(len(x)), 0, 0}
	C.clEnqueueNDRangeKernel(ctx.Queue, kernel, 1,
		nil, &workSize[0], nil, 0, nil, nil)

	ctx.Err = C.clEnqueueReadBuffer(ctx.Queue, dstBuffer, C.cl_bool(1), 0,
		C.size_t(int(unsafe.Sizeof(x[0]))*len(x)), unsafe.Pointer(&y[0]), 0, nil, nil)
	ctx.CheckErr()

	fmt.Println(x, "*", mult, "=", y, "!!")
}
