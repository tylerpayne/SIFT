#include <structs/Core.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#ifndef _MATRIX_
#define _MATRIX_

int VERBOSITY = 0;

typedef struct Matrix Matrix;

typedef void (*voidFunc)();
typedef float (*mp2Func)(Matrix*, Point2);
typedef float* (*mrectFunc)(Matrix*, Rect);
typedef void (*mp2fFunc)(Matrix*, Point2, float);
typedef void (*mrectpfFunc)(Matrix*, Rect, float*);
typedef void (*mFunc)(Matrix*);

struct Matrix
{
  Shape shape;

  float* hostPtr;
  float* devicePtr;
  BOOL isHostSide;
  int T;

  mFunc toDevice;
  mFunc toHost;

  voidFunc transpose;

  mp2Func getElement;
  mrectFunc getRegion;
  mp2fFunc setElement;
  mrectpfFunc setRegion;

  mFunc free;
};

#endif
