package cl

import (
	"math/rand"
	"reflect"
	"testing"
	"unsafe"
)

const sourceCode = `
__kernel void SAXPY (__global float* x, __global float* y, float a)
{
    const int i = get_global_id(0);

    y[i] += a * x[i];
}
`

func TestGetPlatforms(t *testing.T) {
	if num := GetPlatformCount(); num <= 0 {
		t.Error("PlatformCount is", num)
	}
	if platforms := GetPlatformIDs(); platforms == nil || len(platforms) == 0 {
		t.Error("No Platforms found")
	}
}

func TestGetDevices(t *testing.T) {
	if num := GetDeviceCount(); num <= 0 {
		t.Error("DeviceCount is", num)
	}
	if devices := GetDeviceIDs(); devices == nil || len(devices) == 0 {
		t.Error("No Devices found")
	}
}

func TestGetContext(t *testing.T) {
	c := CreateContext()
	c.CheckErr()
}

// Test a simple a*X = Y scalar multiplication
func TestSimpleSAXPY(t *testing.T) {
	num := 16

	// initialize test data
	var mult float32 = rand.Float32()

	x := make([]float32, num)
	y := make([]float32, num)
	expectedY := make([]float32, num)
	for i := range x {
		x[i] = rand.Float32()
		expectedY[i] = x[i] * mult
	}

	// create a Context
	ctx := CreateContext()
	// create our Buffers, the data gets copied
	buffer := ctx.CreateBuffer(x)
	dstBuffer := ctx.CreateBuffer(y)

	// compile the source code, using the method SAXPY as entry point
	kernel := ctx.CompileProgram(sourceCode, "SAXPY")

	// set the arguments
	kernel.SetArg(0, buffer)
	kernel.SetArg(1, dstBuffer)
	kernel.SetArg(2, mult)

	// define work dimension
	workSize := []Size_t{Size_t(num)}
	// start working
	kernel.Enqueue(nil, workSize, nil, nil)

	// read back the data, with blocking set to true
	dstBuffer.EnqueueRead(unsafe.Pointer(&y[0]), unsafe.Sizeof(y[0]), len(y), true, nil)

	// check if it's correct
	if !reflect.DeepEqual(y, expectedY) {
		t.Error("expected ", expectedY, "- got", y)
	}
}
