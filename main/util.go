package main

func max(v1 int, val ...int) int {
	v := v1
	for _, a := range val {
		if a > v {
			v = a
		}
	}
	return v
}
