#include <stdlib.h>
#include "Matrix.h"

typedef struct MatrixUtil MatrixUtil;

typedef void (*m1m2rFunc)(MatrixUtil* , Matrix *, Matrix *, Matrix *);
typedef int (*m1m2Func)(MatrixUtil* , Matrix *, Matrix *);
typedef void (*m1Func)(MatrixUtil* , Matrix *, Matrix*);
typedef Matrix* (*mTFunc)(MatrixUtil* , Matrix *);
typedef Matrix* (*mSFunc)(MatrixUtil* , Matrix *, Matrix*);
typedef void (*m1fFunc)(MatrixUtil* , Matrix *, float, Matrix*);
typedef void (*mrFunc)(MatrixUtil* , Matrix*, Matrix*);
typedef Matrix* (*newMatrixFunc)(int,int);
typedef Matrix* (*newMatrixWithFloatFunc)(float*,int,int);
typedef void (*prettyPrintFunc)(MatrixUtil*,Matrix*,char*);
typedef void (*syncMatrixFunc)(MatrixUtil* , Matrix*);

struct MatrixUtil
{
  int verbosity;
  int deviceId;
  cudaStream_t stream;
  cublasHandle_t cublasHandle;
  newMatrixFunc newEmptyMatrix;
  newMatrixWithFloatFunc newMatrix;
  syncMatrixFunc sync;
  m1Func downsample;
  m1m2rFunc add;
  m1m2rFunc subtract;
  m1m2rFunc multiply;
  m1fFunc multiplyConst;
  m1m2rFunc divide;
  m1fFunc divideConst;
  m1Func sqrt;
  m1m2Func isEqual;
  m1Func arctan;
  m1Func exp;
  m1Func log;
  m1fFunc pow;
  m1Func ceil;
  m1Func floor;
  m1Func abs;
  mTFunc transpose;
  m1m2rFunc dot;
  m1m2rFunc cross;
  mTFunc inv;
  mSFunc solve;
  mSFunc lstsq;
  m1m2rFunc convolve;
  prettyPrintFunc pprint;
};

MatrixUtil* GetCUDAMatrixUtil(int device);
MatrixUtil* GetPrimitiveMatrixUtil();
MatrixUtil* GetMetalMatrixUtil();
