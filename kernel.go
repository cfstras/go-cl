package cl

//#include "include.h"
import "C"

import (
	"fmt"
	"reflect"
	"unsafe"
)

/*
Enqueues a Kernel to be executed. Be sure to set any needed arguments with
Kernel.SetArg() in preparation.

globalWorkOffset, globalWorkSize and localWorkSize are vectors. globalWorkOffset
and localWorkSize can be nil -- otherwise they have have the same dimensions as
globalWorkSize.

The parameter globalWorkOffset can be used to specify an offset in the
globalWorkSize.

The parameter globalWorkSize specifies the total working size.

The parameter localWorkSize can be used to specify the working group sizes.

The parameter eventWaitList, if not nil, specifies a list of events to wait for before
starting to read data.

The returned event can be used to queue other operations after this one.
*/
func (k *Kernel) Enqueue(globalWorkOffset, globalWorkSize, localWorkSize []Size_t,
	eventWaitList []Event) *Event {
	if globalWorkSize == nil {
		panic("globalWorkSize cannot be nil")
	}
	dim := len(globalWorkSize)
	globPtr := &globalWorkSize[0]

	var offPtr, locPtr *Size_t
	if globalWorkOffset != nil {
		offPtr = &globalWorkOffset[0]
		if len(globalWorkOffset) != dim {
			panic("globalWorkOffset has to have the same dimension as globalWorkSize")
		}
	}
	if localWorkSize != nil {
		locPtr = (*Size_t)(&localWorkSize[0])
		if len(localWorkSize) != dim {
			panic("localWorkSize has to have the same dimension as globalWorkSize")
		}
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

/*
Sets an argument for this kernel.

The parameter pos specifies the position, counting from 0.

Currently supported types: Buffer, float32/64, int32/64.
*/
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
	case int32:
		ptr = unsafe.Pointer(&t)
		size = unsafe.Sizeof(t)
	case int64:
		ptr = unsafe.Pointer(&t)
		size = unsafe.Sizeof(t)
	default:
		panic(fmt.Sprint("Type ", reflect.TypeOf(t), " not supported"))
	}
	C.clSetKernelArg(k.K, C.cl_uint(pos), C.size_t(size), ptr)
}
