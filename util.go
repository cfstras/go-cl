package cl

//#include "include.h"
import "C"

import (
	"fmt"
)

func max(v1 int, val ...int) int {
	v := v1
	for _, a := range val {
		if a > v {
			v = a
		}
	}
	return v
}

// Panics on errors
func (c *Context) CheckErr() {
	if c.Err != C.CL_SUCCESS {
		panic(fmt.Sprint("CL error", c.Err, "!"))
	}
}
