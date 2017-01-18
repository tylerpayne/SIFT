#include "Matrix.h"

typedef struct MatrixUtil MatrixUtil;

typedef void (*m1m2rFunc)(MatrixUtil* , Matrix *, Matrix *, Matrix *);
typedef int (*m1m2Func)(MatrixUtil* , Matrix *, Matrix *);
typedef int* (*rintm1Func)(MatrixUtil* , Matrix *);
typedef float (*fm1m2Func)(MatrixUtil* , Matrix *, Matrix *);
typedef void (*m1Func)(MatrixUtil* , Matrix *, Matrix*);
typedef Matrix* (*mTFunc)(MatrixUtil* , Matrix *);
typedef Matrix* (*mSFunc)(MatrixUtil* , Matrix *, Matrix*);
typedef void (*m1fFunc)(MatrixUtil* , Matrix *, float, Matrix*);
typedef void (*mrFunc)(MatrixUtil* , Matrix*, Matrix*);
typedef Matrix* (*newMatrixFunc)(int,int);
typedef Matrix* (*newMatrixWithFloatFunc)(float*,int,int);
typedef void (*prettyPrintFunc)(MatrixUtil*,Matrix*,char*);
typedef void (*syncMatrixFunc)(MatrixUtil* , Matrix*);
typedef void (*mCopyFunc)(MatrixUtil*,Matrix*,Matrix*,Rect,Point2,Point2);

struct MatrixUtil
{
  int verbosity;
  int deviceId;
  newMatrixFunc newEmptyMatrix;
  newMatrixWithFloatFunc newMatrix;
  m1m2rFunc add;
  m1m2rFunc subtract;
  m1m2rFunc multiply;
  m1fFunc multiplyConst;
  m1m2rFunc divide;
  m1fFunc divideConst;
  rintm1Func minRows;
  m1Func sqrt;
  m1Func transpose;
  fm1m2Func distance;
  m1m2Func isEqual;
  m1Func arctan;
  m1Func exp;
  m1Func log;
  m1fFunc pow;
  m1Func ceil;
  m1Func floor;
  m1Func abs;
  m1m2rFunc dot;
  m1m2rFunc featureDistance;
  m1m2rFunc cross;
  m1Func makeCrossMatrix;
  m1Func inv;
  mSFunc solve;
  mSFunc lstsq;

  mCopyFunc copy;
  prettyPrintFunc pprint;
};

DLLEXPORT MatrixUtil* GetMatrixUtil();
