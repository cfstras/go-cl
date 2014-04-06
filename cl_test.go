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

func TestSimpleSAXPY(t *testing.T) {
	num := 16

	var mult float32 = rand.Float32()

	x := make([]float32, num)
	y := make([]float32, num)
	expectedY := make([]float32, num)
	for i := range x {
		x[i] = rand.Float32()
		expectedY[i] = x[i] * mult
	}

	ctx := CreateContext()
	buffer := ctx.CreateBuffer(x)
	dstBuffer := ctx.CreateBuffer(y)

	kernel := ctx.CompileProgram(sourceCode, "SAXPY")

	kernel.SetArg(0, buffer)
	kernel.SetArg(1, dstBuffer)
	kernel.SetArg(2, mult)

	// start!
	workSize := []Size_t{Size_t(num)}
	kernel.Enqueue(nil, workSize, nil, nil)

	dstBuffer.EnqueueRead(unsafe.Pointer(&y[0]), unsafe.Sizeof(y[0]), len(y), true, nil)

	if !reflect.DeepEqual(y, expectedY) {
		t.Error("expected ", expectedY, "- got", y)
	}
}
