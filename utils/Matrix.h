#include <structs/Core.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

int VERBOSITY = 0;

typedef struct Matrix Matrix;

typedef void (*voidFunc)();
typedef float (*getMatrixElementFunc)(Matrix*, int,int);
typedef float* (*getMatrixRegionFunc)(Matrix*, int,int,int,int);
typedef void (*setMatrixElementFunc)(Matrix*, int, int, float);
typedef void (*setMatrixRegionFunc)(Matrix*, int,int,int,int, float*);
typedef void (*freeMatrixFunc)(Matrix*);

struct Matrix
{
  int* shape;
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
