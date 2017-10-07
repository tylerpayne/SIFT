#ifndef _CORE_H_
#define _CORE_H_

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#define HOST 1
#define DEVICE 0

#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <cublas_v2.h>
#include <curand.h>
#include <npp.h>
#include <nppi.h>
#include <cuda_helpers.h>
//#include <lodepng.h>
//#include <errors.h>
#include <assert.h>
#include <complex.h>

//extern cublasHandle_t _cublasHandle;
//extern curandGenerator_t _curandGenerator;

template<typename K>
class Tuple
{
public:
  std::vector<K> components;
  int length;

  Tuple(std::initializer_list<K> coords)
  {
    components(coords);
    length = components.size();
  }

  int product()
  {
    int retval = components[0];
    for (int i = 1; i < length; i++)
    {
      retval *= components[i];
    }
    return retval;
  }

  int norm2()
  {
    int retval = components[0];
    for (int i = 1; i < length; i++)
    {
      retval += components[i]*components[i];
    }
    return sqrt(retval);
  }

  ~Tuple() {delete components;}
};

template<typename T>
class Rect
{
public:
  Tuple<T> origin;
  Tuple<int> shape;
  Rect(Tuple<T> p, Tuple<int> s) {origin = p; shape = s;}
};

#endif
