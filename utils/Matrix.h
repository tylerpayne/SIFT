#include <stdlib.h>

typedef struct Matrix Matrix;

typedef float (*getMatrixElementFunc)(Matrix*, int,int);
typedef float* (*getMatrixRegionFunc)(Matrix*, int,int,int,int);
typedef void (*setMatrixElementFunc)(Matrix*, int, int, float);
typedef void (*setPrimMatrixRegionFunc)(Matrix*, int,int,int,int, float*);

struct Matrix
{
  int* shape;
  void* nativePtr;
  void* devicePtr;
  getMatrixElementFunc getElement;
  getMatrixRegionFunc getRegion;
  setMatrixElementFunc setElement;
  setPrimMatrixRegionFunc setRegion;
};

int IDX2C(int i, int j, int td)
{
  return (i*td)+j;
}
