#include <stdlib.h>
#include "Matrix.h"

typedef struct MatrixUtil MatrixUtil;

typedef void (*m1m2rFunc)(Matrix *, Matrix *, Matrix *);
typedef int (*m1m2Func)(Matrix *, Matrix *);
typedef void (*m1Func)(Matrix *, Matrix*);
typedef Matrix* (*mTFunc)(Matrix *);
typedef Matrix* (*mSFunc)(Matrix *, Matrix*);
typedef void (*m1fFunc)(Matrix *, float, Matrix*);
typedef void (*mrFunc)(Matrix*, Matrix*);
typedef Matrix* (*newMatrixFunc)(int,int);
typedef Matrix* (*newMatrixWithFloatFunc)(float*,int,int);
typedef void (*prettyPrintFunc)(Matrix*,char*);
typedef void (*syncMatrixFunc)(Matrix*);

struct MatrixUtil
{
  int verbosity;
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

MatrixUtil* GetCUDAMatrixUtil();
MatrixUtil* GetPrimitiveMatrixUtil();
MatrixUtil* GetMetalMatrixUtil();
