#ifndef _CORE_H_
#define _CORE_H_

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#define HOST 1
#define DEVICE 0

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <cublas_v2.h>
#include <curand.h>
#include <nppi.h>
#include <lodepng.h>
#include <errors.h>
#include <assert.h>

//extern cublasHandle_t _cublasHandle;
//extern curandGenerator_t _curandGenerator;

typedef struct {
  int x, y;
} Point2;

typedef struct {
  int x, y, z;
} Point3;

typedef struct {
  int x, y, z, w;
} Point4;

typedef struct {
  float x, y;
} Point2f;

typedef struct {
  float x, y, z;
} Point3f;

typedef struct {
  float x, y, z, w;
} Point4f;

typedef struct {
  int width, height;
} Shape;

typedef struct {
  Point2 origin;
  Shape shape;
} Rect;

typedef enum {
  TRUE=1,
  FALSE=0
} BOOL;

enum keyType {
  INT = 0,
  FLOAT = 1,
  STRING = 2,
};

typedef struct {
  int ival;
  float fval;
  char* sval;
  int type;
} Key;

#endif
