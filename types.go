package cl

//#include "include.h"
import "C"

type Size_t C.size_t

// Represents an OpenCL context and a queue.
type Context struct {
	// the abstract context identifier
	C C.cl_context
	// the abstract queue identifier
	Queue C.cl_command_queue

	// Used to store the last error code.
	// Any methods using OpenCL methods should always use this field.
	Err C.cl_int
}

// Represents an OpenCL Buffer object.
type Buffer struct {
	// the abstract Buffer identifier
	B C.cl_mem
	// our context
	Context *Context
}

type Event C.cl_event

// Represents an OpenCL compiled Kernel
type Kernel struct {
	// the abstract Kernel identifier
	K C.cl_kernel
	// our context
	Context *Context
}
