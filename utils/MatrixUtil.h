#include <stdlib.h>

#include <stdlib.h>

typedef struct sMatrix
{
  int* shape;
  void* nativePtr;

} Matrix;

typedef void (*m1m2Func)(Matrix *, Matrix *);
typedef void (*m1Func)(Matrix *);
typedef void (*m1fFunc)(Matrix *, float);
typedef Matrix* (*mrFunc)(Matrix*);

typedef struct sMatrixUtil
{
  m1m2Func add;
  m1m2Func multiply;
  m1m2Func divide;
  m1Func sqrt;
  m1m2Func isEqual;
  m1Func arctan;
  m1Func arctan2;
  m1Func exp;
  m1Func log;
  m1fFunc pow;
  m1fFunc ceil;
  m1fFunc floor;
  m1Func abs;
  m1fFunc mod;
  mrFunc transpose;
  m1m2Func dot;
  m1m2Func cross;
  mrFunc inv;
  
} MatrixUtil;

Matrix* NewMatrix(float* f, int* shape);

MatrixUtil* GetCUDAMatrixUtil();
MatrixUtil* GetPrimitiveMatrixUtil();
MatrixUtil* GetMetalMatrixUtil();
