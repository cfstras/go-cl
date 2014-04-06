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
	"reflect"
	"unsafe"
)

type Context struct {
	C     C.cl_context
	Queue C.cl_command_queue

	NumDevices C.cl_uint
	DeviceIDs  []C.cl_device_id

	Err C.cl_int
}

type Buffer C.cl_mem
type Event C.cl_event

type Kernel struct {
	K       C.cl_kernel
	Context *Context
}

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

	c.GetInfo()

	c.C = C.clCreateContext(&contextProps[0], 1, &c.DeviceIDs[0], nil, nil,
		&c.Err)
	c.CheckErr()

	c.Queue = C.clCreateCommandQueue(c.C, c.DeviceIDs[0], 0, &c.Err)
	c.CheckErr()

	return c
}

func (c *Context) GetInfo() {
	l := 128
	mem := C.malloc(C.size_t(unsafe.Sizeof(C.char(0))) * C.size_t(l))
	defer C.free(mem)
	c.Err = C.clGetDeviceInfo(c.DeviceIDs[0], C.CL_DEVICE_NAME, C.size_t(l), mem, nil)
	name := C.GoString((*C.char)(mem))
	fmt.Println("Name:", name)
}

func (c *Context) CreateBuffer(data []float32) Buffer {
	var buffer C.cl_mem
	buffer = C.clCreateBuffer(c.C, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(int(unsafe.Sizeof(data[0]))*len(data)), unsafe.Pointer(&data[0]),
		&c.Err)
	c.CheckErr()
	return Buffer(buffer)
}

func (c *Context) CompileProgram(code, mainFunction string) *Kernel {
	k := Kernel{Context: c}
	str := C.CString(sourceCode)
	defer C.free(unsafe.Pointer(str))
	prog := C.clCreateProgramWithSource(c.C, 1, &str, nil, &c.Err)
	c.CheckErr()

	C.clBuildProgram(prog, c.NumDevices, &c.DeviceIDs[0], nil, nil, nil)
	c.CheckErr()

	mainFunc := C.CString(mainFunction)
	defer C.free(unsafe.Pointer(mainFunc))
	kernel := C.clCreateKernel(prog, mainFunc, &c.Err)
	c.CheckErr()
	k.K = kernel
	return &k
}

func (k *Kernel) Enqueue(globalWorkOffset, globalWorkSize, localWorkSize []C.size_t,
	eventWaitList []Event) *Event {
	var dimOff, dimGlob, dimLoc int
	var offPtr, globPtr, locPtr *C.size_t
	if globalWorkOffset != nil {
		dimOff = len(globalWorkOffset)
		offPtr = &globalWorkOffset[0]
	}
	if globalWorkSize != nil {
		dimGlob = len(globalWorkSize)
		globPtr = &globalWorkSize[0]
	}
	if localWorkSize != nil {
		dimLoc = len(localWorkSize)
		locPtr = &localWorkSize[0]
	}
	dim := max(dimOff, dimGlob, dimLoc)
	if (dimOff != 0 && dimOff != dim) ||
		(dimGlob != 0 && dimGlob != dim) ||
		(dimLoc != 0 && dimLoc != dim) {
		panic("globalWorkOffset, globalWorkSize and localWorkSize have to be nil or have the same length!")
	}

	var evPtr *Event
	var evLen int
	if eventWaitList != nil {
		evPtr = &eventWaitList[0]
		evLen = len(eventWaitList)
	}
	var event Event

	k.Context.Err = C.clEnqueueNDRangeKernel(k.Context.Queue, k.K, C.cl_uint(dim),
		offPtr, globPtr, locPtr, C.cl_uint(evLen), (*C.cl_event)(evPtr), ((*C.cl_event)(&event)))
	k.Context.CheckErr()
	return &event
}

func (k *Kernel) SetArg(pos int, value interface{}) {
	var ptr unsafe.Pointer
	var size uintptr
	switch t := value.(type) {
	case Buffer:
		ptr = unsafe.Pointer(&t)
		size = unsafe.Sizeof(t)
	case float32:
		ptr = unsafe.Pointer(&t)
		size = unsafe.Sizeof(t)
	case float64:
		ptr = unsafe.Pointer(&t)
		size = unsafe.Sizeof(t)
	case *float32:
		ptr = unsafe.Pointer(t)
		size = unsafe.Sizeof(*t)
	case *float64:
		ptr = unsafe.Pointer(t)
		size = unsafe.Sizeof(*t)
	case *Buffer:
		ptr = unsafe.Pointer(t)
		size = unsafe.Sizeof(*t)
	default:
		panic(fmt.Sprint("Type ", reflect.TypeOf(t), " not supported"))
	}
	C.clSetKernelArg(k.K, C.cl_uint(pos), C.size_t(size), ptr)
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
	var mult float32 = 2.0
	y := make([]float32, 4)
	buffer := ctx.CreateBuffer(x)
	dstBuffer := ctx.CreateBuffer(y)

	kernel := ctx.CompileProgram(sourceCode, "SAXPY")

	kernel.SetArg(0, buffer)
	kernel.SetArg(1, dstBuffer)
	kernel.SetArg(2, mult)

	// start!
	workSize := []C.size_t{C.size_t(len(x))}
	kernel.Enqueue(nil, workSize, nil, nil)

	ctx.Err = C.clEnqueueReadBuffer(ctx.Queue, dstBuffer, C.cl_bool(1), 0,
		C.size_t(int(unsafe.Sizeof(x[0]))*len(x)), unsafe.Pointer(&y[0]), 0, nil, nil)
	ctx.CheckErr()

	fmt.Println(x, "*", mult, "=", y)
}
