#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <core.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Matrix Matrix;
typedef struct CMatrix CMatrix;

struct Matrix
{
  BOOL isHostSide, T;
  float *host_ptr, *dev_ptr;
  Shape shape;
};

struct CMatrix
{
  BOOL isHostSide, T;
  __complex__ float *host_ptr, *dev_ptr;
  Shape shape;
};

#include <matrix_funcs.h>

#ifdef __cplusplus
}
#endif
#endif
