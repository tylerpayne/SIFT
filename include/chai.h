#pragma once

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#define HOST 1
#define DEVICE 0

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>
#include <npp.h>
#include <nppi.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <vector_types.h>
//#include <lodepng.h>
//#include <errors.h>
#include <assert.h>
#include <complex.h>


#include <tuple.h>
#include <matrix.h>
#include <arith.h>
#include <chai_cuda.h>
#include <kernels.h>


namespace chai {

int idx2c(int i, int j, tuple<int> shape);
int idx2c(tuple<int> index, tuple<int> shape);
tuple<int> c2idx(int i, tuple<int> shape);
void make_launch_parameters(tuple<int> shape, int dim, dim3 *bdim, dim3 *gdim);

}
