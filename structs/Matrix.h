#include <structs/Core.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#ifndef _MATRIX_
#define _MATRIX_

int VERBOSITY = 0;

typedef struct Matrix Matrix;

typedef void (*voidFunc)();
typedef float (*getMatrixElementFunc)(Matrix*, Point2);
typedef float* (*getMatrixRegionFunc)(Matrix*, Rect);
typedef void (*setMatrixElementFunc)(Matrix*, Point2, float);
typedef void (*setMatrixRegionFunc)(Matrix*, Rect, float*);
typedef void (*freeMatrixFunc)(Matrix*);

struct Matrix
{
  Shape shape;
  float* hostPtr;
  float* devicePtr;
  int isHostSide;
  int T;
  voidFunc transpose;
  getMatrixElementFunc getElement;
  getMatrixRegionFunc getRegion;
  setMatrixElementFunc setElement;
  setMatrixRegionFunc setRegion;
  freeMatrixFunc free;
};

#endif
