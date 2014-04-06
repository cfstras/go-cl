
GOPATH := $(CURDIR)
export GOPATH

all: build test

.PHONY: build test
build:
	go build github.com/cfstras/go-cl

test:
	go test github.com/cfstras/go-cl
