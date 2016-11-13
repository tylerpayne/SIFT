#include <stdlib.h>
#include "MatrixUtil.h"

typedef Matrix* (*getRegionFunc)(int,int,int,int);

typedef struct sImage
{
  Matrix* data;
  getRegionFunc getRegion;
  
} Image;
