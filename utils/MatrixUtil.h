#include <stdlib.h>
#include "Matrix.h"

typedef Matrix* (*m1m2rFunc)(Matrix *, Matrix *);
typedef int (*m1m2Func)(Matrix *, Matrix *);
typedef Matrix* (*m1Func)(Matrix *);
typedef Matrix* (*m1fFunc)(Matrix *, float);
typedef Matrix* (*mrFunc)(Matrix*);
typedef Matrix* (*newMatrixFunc)(int,int);
typedef Matrix* (*newMatrixWithFloatFunc)(float*,int,int);

typedef struct MatrixUtil MatrixUtil;

struct MatrixUtil
{
  newMatrixFunc newEmptyMatrix;
  newMatrixWithFloatFunc newMatrix;
  m1m2rFunc add;
  m1m2rFunc multiply;
  m1m2rFunc divide;
  m1Func sqrt;
  m1m2Func isEqual;
  m1Func arctan;
  m1Func exp;
  m1Func log;
  m1fFunc pow;
  m1Func ceil;
  m1Func floor;
  m1Func abs;
  mrFunc transpose;
  m1m2rFunc dot;
  m1m2rFunc cross;
  mrFunc inv;
};

MatrixUtil* GetCUDAMatrixUtil();
MatrixUtil* GetPrimitiveMatrixUtil();
MatrixUtil* GetMetalMatrixUtil();
