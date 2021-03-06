package cl

//#include "include.h"
import "C"

import (
	"unsafe"
)

// Creates a context and a queue on the first device on the first platform
func CreateContext() *Context {
	c := &Context{}

	platformIDs := GetPlatformIDs()
	deviceIDs := GetDeviceIDs()

	// make context
	contextProps := []C.cl_context_properties{
		C.CL_CONTEXT_PLATFORM,
		*(*C.cl_context_properties)(unsafe.Pointer(&platformIDs[0])),
		0,
	}

	c.C = C.clCreateContext(&contextProps[0], 1, &deviceIDs[0], nil, nil,
		&c.Err)
	c.CheckErr()

	c.Queue = C.clCreateCommandQueue(c.C, deviceIDs[0], 0, &c.Err)
	c.CheckErr()

	return c
}

// Returns the number of OpenCL platforms
func GetPlatformCount() int {
	var platformCount C.cl_uint
	CheckErr(C.clGetPlatformIDs(0, nil, &platformCount))

	return int(platformCount)
}

// Returns a slice of platform IDs
func GetPlatformIDs() []C.cl_platform_id {
	platformCount := GetPlatformCount()
	platformIDs := make([]C.cl_platform_id, platformCount)
	CheckErr(C.clGetPlatformIDs(C.cl_uint(platformCount), &platformIDs[0], nil))

	return platformIDs
}

// Returns the number of OpenCL devices on the first Platform
func GetDeviceCount() int {
	platformIDs := GetPlatformIDs()
	var deviceCount C.cl_uint
	CheckErr(C.clGetDeviceIDs(platformIDs[0], C.CL_DEVICE_TYPE_ALL, 0, nil,
		&deviceCount))

	return int(deviceCount)
}

// Returns a slice of device IDs on the first platform
func GetDeviceIDs() []C.cl_device_id {
	platformIDs := GetPlatformIDs()
	deviceCount := GetDeviceCount()
	deviceIDs := make([]C.cl_device_id, deviceCount)
	CheckErr(C.clGetDeviceIDs(platformIDs[0], C.CL_DEVICE_TYPE_ALL,
		C.cl_uint(deviceCount), &deviceIDs[0], nil))

	return deviceIDs
}

// returns the name of the current device
func (c *Context) GetDeviceName() string {
	l := 128
	mem := C.malloc(C.size_t(unsafe.Sizeof(C.char(0))) * C.size_t(l))
	defer C.free(mem)

	c.Err = C.clGetDeviceInfo(GetDeviceIDs()[0], C.CL_DEVICE_NAME, C.size_t(l), mem, nil)
	name := C.GoString((*C.char)(mem))
	return name
}

// Creates a buffer from a slice of data
func (c *Context) CreateBuffer(data []float32) Buffer {
	buffer := Buffer{Context: c}
	buffer.B = C.clCreateBuffer(c.C, C.CL_MEM_READ_ONLY|C.CL_MEM_COPY_HOST_PTR,
		C.size_t(int(unsafe.Sizeof(data[0]))*len(data)), unsafe.Pointer(&data[0]),
		&c.Err)
	c.CheckErr()
	return buffer
}

// Compiles a program, using mainFunction as entry point, returning a kernel
// object.
func (c *Context) CompileProgram(code, mainFunction string) *Kernel {
	k := Kernel{Context: c}
	str := C.CString(code)
	defer C.free(unsafe.Pointer(str))
	prog := C.clCreateProgramWithSource(c.C, 1, &str, nil, &c.Err)
	c.CheckErr()

	deviceIDs := GetDeviceIDs()
	C.clBuildProgram(prog, C.cl_uint(len(deviceIDs)), &deviceIDs[0], nil, nil, nil)
	c.CheckErr()

	mainFunc := C.CString(mainFunction)
	defer C.free(unsafe.Pointer(mainFunc))
	kernel := C.clCreateKernel(prog, mainFunc, &c.Err)
	c.CheckErr()
	k.K = kernel
	return &k
}
