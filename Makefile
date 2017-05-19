CUDA_PATH=/usr/local/cuda-8.0/bin
CC=$(CUDA_PATH)/nvcc

COMMON_C_SOURCES = $(wildcard common/*.c)
COMMON_CU_SOURCES = $(wildcard common/*.cu)

CURRENT=

INPUT=main.cpp
OUTPUT=out

.PHONY: all compile link clean clean_all

all:
	$(MAKE) compile CURRENT=matrix
	$(MAKE) link CURRENT=matrix
	$(MAKE) clean
	$(MAKE) compile CURRENT=graph
	$(MAKE) link CURRENT=graph
	$(MAKE) clean

compile:
	$(CC) --compiler-options=-fPIC -std=c++11 --device-c -I./include $(COMMON_C_SOURCES) $(COMMON_CU_SOURCES) $(shell ls $(CURRENT)/*.c) $(shell ls $(CURRENT)/*.cu) -lnppc -lcublas -lcurand -lnppi

link:
	$(CC) -shared --compiler-options=-fPIC -std=c++11 -I./src $(shell ls *.o) -o bin/lib$(CURRENT).so

clean_all:
	rm bin/*

clean:
	rm *.o

app:
	$(CC) -std=c++11 -Iinclude $(INPUT) -o $(OUTPUT) -Lbin -lmatrix -lcublas -lnppc -lnppi -lcurand
