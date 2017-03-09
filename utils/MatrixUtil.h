#include <structs/Matrix.h>

#ifndef _MATRIXUTIL_
#define _MATRIXUTIL_

typedef struct MatrixUtil MatrixUtil;

typedef void (*mFunc)(MatrixUtil*, Matrix*);
typedef void (*mmFunc)(MatrixUtil* , Matrix *, Matrix *);
typedef void (*mmmFunc)(MatrixUtil* , Matrix *, Matrix *, Matrix *);

typedef void (*mfmFunc)(MatrixUtil*, Matrix*, float, Matrix*);
typedef void (*mpiFunc)(MatrixUtil*, Matrix*, int*);
typedef void (*mpfFunc)(MatrixUtil*, Matrix*, float*);

typedef void (*mmrectp2p2Func)(MatrixUtil*,Matrix*,Matrix*,Rect,Point2,Point2);

typedef void (*msFunc)(MatrixUtil*,Matrix*,char*);

typedef void* (*shapeFunc)(Shape);
typedef void* (*pfshapeFunc)(float*,Shape);

struct MatrixUtil
{
  shapeFunc newEmptyMatrix;
  pfshapeFunc newMatrix;

  mmmFunc add, subtract, multiply divide, dot, cross;
  mfmFunc addf, subtractf, multiplyf, dividef;

  mmFunc abs, sqrt, cos, sin, tan, acos, asin, atan, exp, log;
  mfmFunc pow;

  mmFunc floor, ceil, transpose;
  mmmFunc isEqual;

  mmFunc inv;
  mmmFunc solve, lstsq;

  mpfFunc max, min;
  mpiFunc argmax, argmin;
  
  mmrectp2p2Func slice, copy;
  msFunc pprint;

  //mmrFunc featureDistance;

};

DLLEXPORT MatrixUtil* GetMatrixUtil();

DLLEXPORT void copyDeviceToDeviceCudaMatrix(Matrix* A, Matrix* B);
DLLEXPORT void copyHostToDeviceCudaMatrix(Matrix* A);
DLLEXPORT void copyDeviceToHostCudaMatrix(Matrix* A);

#endif
