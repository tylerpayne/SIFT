#include <stdlib.h>
#include "MatrixUtil.h"

typedef struct Image Image;
typedef Matrix* (*getRegionFunc)(Image* self,int,int,int,int);
struct Image
{
  Matrix* data;
  getRegionFunc getRegion;
};
