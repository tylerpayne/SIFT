#include <math.h>
#include "MatrixUtil.h"


float getPrimMatrixElementImpl(Matrix* self, int i, int  j)
{
  return ((float*)self->nativePtr)[IDX2C(i,j,self->shape[1])];
}

float* getPrimMatrixRegionImpl(Matrix* self, int* region)
{
  /*
  int* region should have four elements (row,col,height_inrows,width_incol)
  */
  int i = region[0];
  int j = region[1];
  int r = region[2];
  int c = region[3];

  float* data = (float*)malloc(sizeof(float)*r*c);
  int counter = 0;
  for (int z = i; z < i+r; z++)
  {
    for (int y = j; y < j+c; y++)
    {
      data[counter] = self->getElement(self,z,y);
      counter++;
    }
  }
  return data;
}

void setPrimMatrixElementImpl(Matrix* self, int i, int  j, float x)
{
  ((float*)self->nativePtr)[IDX2C(i,j,self->shape[1])] = x;
}

void setPrimMatrixRegionImpl(Matrix* self, int* region, float* data)
{
  /*
  int* region should have four elements (row,col,height_inrows,width_incol)
  */
  int i = region[0];
  int j = region[1];
  int r = region[2];
  int c = region[3];

  int counter = 0;
  for (int z = i; z < i+r; z++)
  {
    for (int y = j; y < j+c; y++)
    {
      self->setElement(self,z,y,data[counter]);
      counter++;
    }
  }
}

Matrix* newEmptyPrimMatrixImpl(int rows, int columns)
{
  Matrix* m = (Matrix*)malloc(sizeof(Matrix));

  float* data = (float*)malloc(sizeof(float)*rows*columns);
  m->nativePtr = (void*)data;

  int* shape = (int*)malloc(sizeof(int)*2);
  shape[0] = rows;
  shape[1] = columns;
  m->shape = shape;

  m->getElement = getPrimMatrixElementImpl;
  m->getRegion = getPrimMatrixRegionImpl;
  m->setElement = setPrimMatrixElementImpl;
  m->setRegion = setPrimMatrixRegionImpl;
  return m;
}

Matrix* newPrimMatrixImpl(float* data, int rows, int columns)
{
  Matrix* m = newEmptyPrimMatrixImpl(rows,columns);
  free(m->nativePtr);
  m->nativePtr = (void*)data;
  return m;
}

Matrix* addPrimMatrixImpl(Matrix* a, Matrix* b)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)+b->getElement(b,i,j));
    }
  }
  return retval;
}

Matrix* multiplyPrimMatrixImpl(Matrix* a, Matrix* b)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)*b->getElement(b,i,j));
    }
  }
  return retval;
}
Matrix* multiplyConstPrimMatrixImpl(Matrix* a, float k)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)*k);
    }
  }
  return retval;
}
Matrix* dividePrimMatrixImpl(Matrix* a, Matrix* b)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)/b->getElement(b,i,j));
    }
  }
  return retval;
}
Matrix* sqrtPrimMatrixImpl(Matrix* a)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,sqrtf(a->getElement(a,i,j)));
    }
  }
  return retval;
}

int isEqualPrimMatrixImpl(Matrix* a, Matrix* b)
{
  int retval = 1;
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      if (a->getElement(a,i,j) != b->getElement(b,i,j))
      {
        retval = 0;
      }
    }
  }
  return retval;
}

Matrix* arctanPrimMatrixImpl(Matrix* a)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,atanf(a->getElement(a,i,j)));
    }
  }
  return retval;
}

Matrix* expPrimMatrixImpl(Matrix* a)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,expf(a->getElement(a,i,j)));
    }
  }
  return retval;
}

Matrix* logPrimMatrixImpl(Matrix* a)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,logf(a->getElement(a,i,j)));
    }
  }
  return retval;
}

Matrix* powPrimMatrixImpl(Matrix* a, float k)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,powf(a->getElement(a,i,j),k));
    }
  }
  return retval;
}

Matrix* ceilPrimMatrixImpl(Matrix* a)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,ceilf(a->getElement(a,i,j)));
    }
  }
  return retval;
}

Matrix* floorPrimMatrixImpl(Matrix* a)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,floorf(a->getElement(a,i,j)));
    }
  }
  return retval;
}

Matrix* absPrimMatrixImpl(Matrix* a)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],a->shape[1]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,fabs(a->getElement(a,i,j)));
    }
  }
  return retval;
}

Matrix* transposePrimMatrixImpl(Matrix* a)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[1],a->shape[0]);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,j,i,a->getElement(a,i,j));
    }
  }
  return retval;
}

float vecDotPrimMatrixImpl(float* a, int dA, float* b, int dB)
{
  float retval;
  for (int i = 0; i < dA; i++)
  {
    retval += a[i]*b[i];
  }
  return retval;
}

Matrix* dotPrimMatrixImpl(Matrix* a, Matrix* b)
{
  Matrix* retval = newEmptyPrimMatrixImpl(a->shape[0],b->shape[1]);
  Matrix* bT = transposePrimMatrixImpl(b);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < b->shape[1]; j++)
    {
      retval->setElement(retval,i,j,vecDotPrimMatrixImpl(&((float*)a->nativePtr)[IDX2C(i,0,a->shape[1])],a->shape[0],&((float*)bT->nativePtr)[IDX2C(j,0,bT->shape[1])],bT->shape[0]));
    }
  }
  return retval;
}
/*
Matrix* crossPrimMatrixImpl(Matrix* a, Matrix* b)
{

}

Matrix* invPrimMatrixImpl(Matrix* a)
{

}

Matrix* solvePrimMatrixImpl(Matrix* A, matrix* y)
{

}

Matrix* llsqPrimMatrixImpl(Matrix* X, Matrix* y)
{

}
*/



MatrixUtil* GetPrimitiveMatrixUtil()
{
  MatrixUtil* primitiveMatrixUtil = (MatrixUtil*)malloc(sizeof(MatrixUtil));

  primitiveMatrixUtil->newEmptyMatrix = newEmptyPrimMatrixImpl;
  primitiveMatrixUtil->newMatrix = newPrimMatrixImpl;
  primitiveMatrixUtil->add = addPrimMatrixImpl;
  primitiveMatrixUtil->multiply = multiplyPrimMatrixImpl;
  primitiveMatrixUtil->divide = dividePrimMatrixImpl;
  primitiveMatrixUtil->sqrt = sqrtPrimMatrixImpl;
  primitiveMatrixUtil->isEqual = isEqualPrimMatrixImpl;
  primitiveMatrixUtil->arctan = arctanPrimMatrixImpl;
  primitiveMatrixUtil->exp = expPrimMatrixImpl;
  primitiveMatrixUtil->log = logPrimMatrixImpl;
  primitiveMatrixUtil->pow = powPrimMatrixImpl;
  primitiveMatrixUtil->ceil = ceilPrimMatrixImpl;
  primitiveMatrixUtil->floor = floorPrimMatrixImpl;
  primitiveMatrixUtil->abs = absPrimMatrixImpl;
  primitiveMatrixUtil->transpose = transposePrimMatrixImpl;
  primitiveMatrixUtil->dot = dotPrimMatrixImpl;
//CROSS AND INV

  return primitiveMatrixUtil;
}
