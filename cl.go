package cl

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

type Size_t C.size_t

type Context struct {
	C     C.cl_context
	Queue C.cl_command_queue

	NumDevices C.cl_uint
	DeviceIDs  []C.cl_device_id

	Err C.cl_int
}

type Buffer struct {
	B       C.cl_mem
	Context *Context
}

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
	buffer := Buffer{Context: c}
	buffer.B = C.clCreateBuffer(c.C, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(int(unsafe.Sizeof(data[0]))*len(data)), unsafe.Pointer(&data[0]),
		&c.Err)
	c.CheckErr()
	return buffer
}

func (c *Context) CompileProgram(code, mainFunction string) *Kernel {
	k := Kernel{Context: c}
	str := C.CString(code)
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

func (k *Kernel) Enqueue(globalWorkOffset, globalWorkSize, localWorkSize []Size_t,
	eventWaitList []Event) *Event {
	var dimOff, dimGlob, dimLoc int
	var offPtr, globPtr, locPtr *Size_t
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
		locPtr = (*Size_t)(&localWorkSize[0])
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
		(*C.size_t)(offPtr), (*C.size_t)(globPtr), (*C.size_t)(locPtr),
		C.cl_uint(evLen), (*C.cl_event)(evPtr), ((*C.cl_event)(&event)))
	k.Context.CheckErr()
	return &event
}

func (k *Kernel) SetArg(pos int, value interface{}) {
	var ptr unsafe.Pointer
	var size uintptr
	switch t := value.(type) {
	case Buffer:
		ptr = unsafe.Pointer(&t.B)
		size = unsafe.Sizeof(t.B)
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
	default:
		panic(fmt.Sprint("Type ", reflect.TypeOf(t), " not supported"))
	}
	C.clSetKernelArg(k.K, C.cl_uint(pos), C.size_t(size), ptr)
}

func (b *Buffer) EnqueueRead(targetPtr unsafe.Pointer, targetElSize uintptr, targetLen int,
	blocking bool, eventWaitList []Event) Event {
	var block C.cl_bool
	if blocking {
		block = 1
	}

	var evPtr *Event
	var evLen int
	if eventWaitList != nil {
		evPtr = &eventWaitList[0]
		evLen = len(eventWaitList)
	}
	var event Event

	b.Context.Err = C.clEnqueueReadBuffer(b.Context.Queue, b.B, block, 0,
		C.size_t(int(targetElSize)*targetLen), targetPtr,
		C.cl_uint(evLen), (*C.cl_event)(evPtr), (*C.cl_event)(&event))
	b.Context.CheckErr()

	return event
}

func (c *Context) CheckErr() {
	if c.Err != C.CL_SUCCESS {
		fmt.Println("CL error", c.Err, "!")
		panic("CL error")
	}
}
