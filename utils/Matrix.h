#include <stdlib.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

int VERBOSITY = 0;

typedef struct Matrix Matrix;

typedef float (*getMatrixElementFunc)(Matrix*, int,int);
typedef float* (*getMatrixRegionFunc)(Matrix*, int,int,int,int);
typedef void (*setMatrixElementFunc)(Matrix*, int, int, float);
typedef void (*setMatrixRegionFunc)(Matrix*, int,int,int,int, float*);


struct Matrix
{
  int* shape;
  void* nativePtr;
  void* devicePtr;
  int isHostSide;
  getMatrixElementFunc getElement;
  getMatrixRegionFunc getRegion;
  setMatrixElementFunc setElement;
  setMatrixRegionFunc setRegion;
};

int IDX2C(int i, int j, int td)
{
  return (i*td)+j;
}
