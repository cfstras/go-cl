package cl

//#include "include.h"
import "C"

import (
	"unsafe"
)

/*
Reads from a Buffer object to client memory.

targetPtr is a pointer to the host memory, targetElSize the size of a single
element and targetLen is the number of elements to read.

These can be requested like this:
    data := make([]float32, 1234)
    targetPtr := unsafe.Pointer(&data[0])
    targetElSize := unsafe.Sizeof(data[0])
    targetLen := len(data)

The boolean "blocking" specifies whether the method should block until the
action is completed.

The parameter eventWaitList, if not nil, specifies a list of events to wait for before
starting to read data.

The returned event can be used to queue other operations after this one.
*/
func (b *Buffer) EnqueueRead(targetPtr unsafe.Pointer, targetElSize uintptr,
	targetLen int, blocking bool, eventWaitList []Event) *Event {
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

	return &event
}
