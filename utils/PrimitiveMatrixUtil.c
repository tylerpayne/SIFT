#include <math.h>
#include <string.h>
#include "MatrixUtil.h"

float getPrimMatrixElementImpl(Matrix* self, int i, int  j)
{
  return ((float*)self->nativePtr)[IDX2C(i,j,self->shape[1])];
}

float* getPrimMatrixRegionImpl(Matrix* self, int i, int j, int h, int w)
{

  float* data = (float*)malloc(sizeof(float)*h*w);

  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      data[IDX2C(y,x,w)] = self->getElement(self,y+i,x+j);
    }
  }

  return data;
}

void setPrimMatrixElementImpl(Matrix* self, int i, int  j, float x)
{
  ((float*)self->nativePtr)[IDX2C(i,j,self->shape[1])] = x;
}

void setPrimMatrixRegionImpl(Matrix* self, int i, int j, int r, int c, float* data)
{
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

void addPrimMatrixImpl(Matrix* a, Matrix* b, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)+b->getElement(b,i,j));
    }
  }
}

void subtractPrimMatrixImpl(Matrix* a, Matrix* b, Matrix* retval)
{
  /*
  a-b
  */
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j) -  b->getElement(b,i,j));
    }
  }
}

void multiplyPrimMatrixImpl(Matrix* a, Matrix* b, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)*b->getElement(b,i,j));
    }
  }
}
void multiplyConstPrimMatrixImpl(Matrix* a, float k, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)*k);
    }
  }
}

void dividePrimMatrixImpl(Matrix* a, Matrix* b, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)/b->getElement(b,i,j));
    }
  }
}

void divideConstPrimMatrixImpl(Matrix* a, float b, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,a->getElement(a,i,j)/b);
    }
  }
}

void sqrtPrimMatrixImpl(Matrix* a, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,sqrtf(a->getElement(a,i,j)));
    }
  }
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

void arctanPrimMatrixImpl(Matrix* a, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,atanf(a->getElement(a,i,j)));
    }
  }
}

void expPrimMatrixImpl(Matrix* a, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,expf(a->getElement(a,i,j)));
    }
  }
}

void logPrimMatrixImpl(Matrix* a, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,logf(a->getElement(a,i,j)));
    }
  }
}

void powPrimMatrixImpl(Matrix* a, float k, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,powf(a->getElement(a,i,j),k));
    }
  }
}

void ceilPrimMatrixImpl(Matrix* a, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,ceilf(a->getElement(a,i,j)));
    }
  }
}

void floorPrimMatrixImpl(Matrix* a, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,floorf(a->getElement(a,i,j)));
    }
  }
}

void absPrimMatrixImpl(Matrix* a, Matrix* retval)
{
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < a->shape[1]; j++)
    {
      retval->setElement(retval,i,j,fabs(a->getElement(a,i,j)));
    }
  }
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
  float retval = 0;
  for (int i = 0; i < dA; i++)
  {
    retval += a[i]*b[i];
  }
  return retval;
}

void dotPrimMatrixImpl(Matrix* a, Matrix* b, Matrix* retval)
{
  Matrix* bT = transposePrimMatrixImpl(b);
  for (int i = 0; i < a->shape[0]; i++)
  {
    for (int j = 0; j < b->shape[1]; j++)
    {
      retval->setElement(retval,i,j,vecDotPrimMatrixImpl(&((float*)a->nativePtr)[IDX2C(i,0,a->shape[1])],a->shape[0],&((float*)bT->nativePtr)[IDX2C(j,0,bT->shape[1])],bT->shape[0]));
    }
  }
}
/*
Matrix* crossPrimMatrixImpl(Matrix* a, Matrix* b)
{

}
*/
Matrix* invPrimMatrixImpl(Matrix* A)
{
  int height = A->shape[0];
  int width = A->shape[1];
  float* solverData = (float*)malloc(sizeof(float)*height*width);
  memcpy(solverData,A->nativePtr,sizeof(float)*height*width);
  float* resultsData = (float*)malloc(sizeof(float)*height*width);
  for (int j = 0; j < width; j++)
  {
    resultsData[IDX2C(j,j,width)] = 1.0;
  }
  Matrix* solver = newPrimMatrixImpl(solverData,height,width);
  Matrix* results = newPrimMatrixImpl(resultsData,height,width);
  for (int col = 0; col<width-1; col++)
  {
    Matrix* pivot = newPrimMatrixImpl(solver->getRegion(solver,col,0,1,width),1,width);
    float normalize = pivot->getElement(pivot,0,col);
    divideConstPrimMatrixImpl(pivot,normalize,pivot);
    solver->setRegion(solver,col,0,1,width,(float*)pivot->nativePtr);
    Matrix* resultPivot = newPrimMatrixImpl(results->getRegion(results,col,0,1,width),1,width);
    divideConstPrimMatrixImpl(resultPivot,normalize,resultPivot);
    results->setRegion(results,col,0,1,width,(float*)resultPivot->nativePtr);
    for (int row = col+1; row<height; row++)
    {
      Matrix *thisRow = newPrimMatrixImpl(solver->getRegion(solver,row,0,1,width),1,width);
      Matrix* thisResultRow = newPrimMatrixImpl(results->getRegion(results,row,0,1,width),1,width);
      float ratio = thisRow->getElement(thisRow,0,col) / pivot->getElement(pivot,0,col);
      Matrix* scaledPivot = newEmptyPrimMatrixImpl(pivot->shape[0],pivot->shape[1]);
      multiplyConstPrimMatrixImpl(pivot,ratio,scaledPivot);
      Matrix* updatedRow = newEmptyPrimMatrixImpl(thisRow->shape[0],thisRow->shape[1]);
      subtractPrimMatrixImpl(thisRow,scaledPivot,updatedRow);
      Matrix* scaledResult = newEmptyPrimMatrixImpl(resultPivot->shape[0],resultPivot->shape[1]);
      multiplyConstPrimMatrixImpl(resultPivot,ratio,scaledResult);
      Matrix* updatedResult = newEmptyPrimMatrixImpl(thisResultRow->shape[0],thisResultRow->shape[1]);
      subtractPrimMatrixImpl(thisResultRow,scaledResult,updatedResult);
      //pprintPrimMatrixImpl(updatedRow,"updatedRow");
      //pprintPrimMatrixImpl(thisRow,"thisRow");
      results->setRegion(results,row,0,1,width,(float*)updatedResult->nativePtr);
      solver->setRegion(solver,row,0,1,width,(float*)updatedRow->nativePtr);
    }
  }
  //Norm bottom right element
  Matrix* pivot = newPrimMatrixImpl(solver->getRegion(solver,width-1,0,1,width),1,width);
  float normalize = pivot->getElement(pivot,0,width-1);
  divideConstPrimMatrixImpl(pivot,normalize,pivot);
  solver->setRegion(solver,width-1,0,1,width,(float*)pivot->nativePtr);
  Matrix* resultPivot = newPrimMatrixImpl(results->getRegion(results,width-1,0,1,width),1,width);
  divideConstPrimMatrixImpl(resultPivot,normalize,resultPivot);
  results->setRegion(results,width-1,0,1,width,(float*)resultPivot->nativePtr);

  for (int col = width-1; col>=0; col--)
  {
    Matrix* pivot = newPrimMatrixImpl(solver->getRegion(solver,col,0,1,width),1,width);
    Matrix* resultPivot = newPrimMatrixImpl(results->getRegion(results,col,0,1,width),1,width);
    for (int row = col-1; row>=0; row--)
    {
      Matrix *thisRow = newPrimMatrixImpl(solver->getRegion(solver,row,0,1,width),1,width);
      Matrix* thisResultRow = newPrimMatrixImpl(results->getRegion(results,row,0,1,width),1,width);
      float ratio = thisRow->getElement(thisRow,0,col) / pivot->getElement(pivot,0,col);
      Matrix* scaledPivot = newEmptyPrimMatrixImpl(pivot->shape[0],pivot->shape[1]);
      multiplyConstPrimMatrixImpl(pivot,ratio,scaledPivot);
      Matrix* updatedRow = newEmptyPrimMatrixImpl(thisRow->shape[0],thisRow->shape[1]);
      subtractPrimMatrixImpl(thisRow,scaledPivot,updatedRow);
      Matrix* scaledResult = newEmptyPrimMatrixImpl(resultPivot->shape[0],resultPivot->shape[1]);
      multiplyConstPrimMatrixImpl(resultPivot,ratio,scaledResult);
      Matrix* updatedResult = newEmptyPrimMatrixImpl(thisResultRow->shape[0],thisResultRow->shape[1]);
      subtractPrimMatrixImpl(thisResultRow,scaledResult,updatedResult);
      results->setRegion(results,row,0,1,width,(float*)updatedResult->nativePtr);
      solver->setRegion(solver,row,0,1,width,(float*)updatedRow->nativePtr);
    }
  }
  for (int col = width-1; col>=0; col--)
  {
    Matrix* pivot = newPrimMatrixImpl(solver->getRegion(solver,col,0,1,width),1,width);
    Matrix* resultPivot = newPrimMatrixImpl(results->getRegion(results,col,0,1,width),1,width);
    float divisor = pivot->getElement(pivot,0,col);
    divideConstPrimMatrixImpl(pivot,divisor,pivot);
    divideConstPrimMatrixImpl(resultPivot,divisor,resultPivot);
    results->setRegion(results,col,0,1,width,(float*)resultPivot->nativePtr);
    solver->setRegion(solver,col,0,1,width,(float*)pivot->nativePtr);
  }
  return results;
}

Matrix* solvePrimMatrixImpl(Matrix* A, Matrix* y)
{
  int height = A->shape[0];
  int width = A->shape[1];
  float* solverData = (float*)malloc(sizeof(float)*height*width);
  memcpy(solverData,A->nativePtr,sizeof(float)*height*width);
  float* resultsData = (float*)malloc(sizeof(float)*height);
  memcpy(resultsData,y->nativePtr,sizeof(float)*height);
  Matrix* solver = newPrimMatrixImpl(solverData,height,width);
  Matrix* results = newPrimMatrixImpl(resultsData,height,1);

  for (int col = 0; col<width-1; col++)
  {
    Matrix* pivot = newPrimMatrixImpl(solver->getRegion(solver,col,0,1,width),1,width);
    float resultPivot = results->getElement(results,col,0);
    for (int row = col+1; row<height; row++)
    {
      Matrix *thisRow = newPrimMatrixImpl(solver->getRegion(solver,row,0,1,width),1,width);
      float ratio = thisRow->getElement(thisRow,0,col) / pivot->getElement(pivot,0,col);
      Matrix* scaledPivot = newEmptyPrimMatrixImpl(pivot->shape[0],pivot->shape[1]);
      multiplyConstPrimMatrixImpl(pivot,ratio,scaledPivot);
      Matrix* updatedRow = newEmptyPrimMatrixImpl(thisRow->shape[0],thisRow->shape[1]);
      subtractPrimMatrixImpl(thisRow,scaledPivot,updatedRow);
      float scaledResult = resultPivot*ratio;
      float updatedResult = results->getElement(results,row,0) - scaledResult;
      results->setElement(results,row,0,updatedResult);
      solver->setRegion(solver,row,0,1,width,(float*)updatedRow->nativePtr);
    }
  }
  for (int col = width-1; col >= 0; col--)
  {
    float thisUnknown = solver->getElement(solver,col,col);
    float thisResult = results->getElement(results,col,0);
    float updatedResult = thisResult/thisUnknown;
    solver->setElement(solver,col,col,1.0);
    results->setElement(results,col,0,updatedResult);
    for (int row = col-1; row>=0; row--)
    {
      float substitute = solver->getElement(solver, row, col);
      substitute = substitute*updatedResult;
      float substiuteResult = results->getElement(results,row,0);
      substiuteResult = substiuteResult - substitute;

      solver->setElement(solver,row,col,0.0);
      results->setElement(results,row,0,substiuteResult);
    }
  }
  return results;
}
void pprintPrimMatrixImpl(Matrix* A, char* label)
{
  printf("\n\n################################################");
  printf("\n%s:\n\n",label);
  for (int i = 0; i < A->shape[0]; i++)
  {
    for (int j = 0;j< A->shape[1];j++)
    {
      printf("[ %f ]",A->getElement(A,i,j));
    }
    //printf("|  %f",y->getElement(y,i,0));
    printf("\n");
  }
  printf("\n################################################\n\n");
}

Matrix* lstsqPrimMatrixImpl(Matrix* A, Matrix* b)
{
  int observations = A->shape[0];
  int parameters = A->shape[1];
  float* XData = (float*)malloc(sizeof(float)*observations*parameters);
  memcpy(XData,A->nativePtr,sizeof(float)*observations*parameters);
  float* yData = (float*)malloc(sizeof(float)*observations);
  memcpy(yData,b->nativePtr,sizeof(float)*observations);
  Matrix* X = newPrimMatrixImpl(XData,observations,parameters);
  Matrix* y = newPrimMatrixImpl(yData,observations,1);
  Matrix* XT = transposePrimMatrixImpl(X);
  pprintPrimMatrixImpl(X,"X");
  pprintPrimMatrixImpl(XT,"XT");
  Matrix* Gramian = newEmptyPrimMatrixImpl(XT->shape[0],X->shape[1]);
  dotPrimMatrixImpl(XT,X,Gramian);
  pprintPrimMatrixImpl(Gramian,"Gramian");
  Gramian = invPrimMatrixImpl(Gramian);
  pprintPrimMatrixImpl(Gramian,"invGramian");
  Matrix* XTy = newEmptyPrimMatrixImpl(XT->shape[0],y->shape[1]);
  dotPrimMatrixImpl(XT,y,XTy);
  pprintPrimMatrixImpl(XTy,"XTy");
  Matrix* retval = newEmptyPrimMatrixImpl(Gramian->shape[0],XTy->shape[1]);
  dotPrimMatrixImpl(Gramian,XTy,retval);
  return retval;
}



void convolvePrimMatrixImpl(Matrix* im, Matrix* kernel, Matrix* retval)
{
  int iw = im->shape[0];
  int ih = im->shape[1];
  int kw = kernel->shape[0];
  int kh = kernel->shape[1];

  int radius = kw/2;

  for (int i = 0; i < iw; i++)
  {
    for (int j = 0; j < ih; j++)
    {
      for (int ki = 0; ki< kw; ki++)
      {
        for (int kj = 0; kj<kh ;kj++)
        {
          if (i+(ki-radius) > 0 && i+(ki-radius) < iw && j+(kj-radius) > 0 && j+(kj-radius) < ih)
          {
            float val = kernel->getElement(kernel,ki,kj) * im->getElement(im,i+(ki-radius),j+(kj-radius));
            ((float*)retval->nativePtr)[IDX2C(i,j,ih)] += val;
          }
        }
      }
    }
  }
}


void syncPrimMatrixImpl(Matrix* m)
{
  return;
}


MatrixUtil* GetPrimitiveMatrixUtil()
{
  MatrixUtil* primitiveMatrixUtil = (MatrixUtil*)malloc(sizeof(MatrixUtil));

  primitiveMatrixUtil->newEmptyMatrix = newEmptyPrimMatrixImpl;
  primitiveMatrixUtil->newMatrix = newPrimMatrixImpl;
  primitiveMatrixUtil->sync = syncPrimMatrixImpl;
  primitiveMatrixUtil->pprint = pprintPrimMatrixImpl;
  primitiveMatrixUtil->add = addPrimMatrixImpl;
  primitiveMatrixUtil->subtract = subtractPrimMatrixImpl;
  primitiveMatrixUtil->multiply = multiplyPrimMatrixImpl;
  primitiveMatrixUtil->multiplyConst = multiplyConstPrimMatrixImpl;
  primitiveMatrixUtil->divide = dividePrimMatrixImpl;
  primitiveMatrixUtil->divideConst = divideConstPrimMatrixImpl;
  primitiveMatrixUtil->sqrt = sqrtPrimMatrixImpl;
  primitiveMatrixUtil->isEqual = isEqualPrimMatrixImpl;
  primitiveMatrixUtil->arctan = arctanPrimMatrixImpl;
  primitiveMatrixUtil->exp = expPrimMatrixImpl;
  primitiveMatrixUtil->log = logPrimMatrixImpl;
  primitiveMatrixUtil->pow = powPrimMatrixImpl;
  primitiveMatrixUtil->ceil = ceilPrimMatrixImpl;
  primitiveMatrixUtil->floor = floorPrimMatrixImpl;
  primitiveMatrixUtil->abs = absPrimMatrixImpl;
  //
  primitiveMatrixUtil->transpose = transposePrimMatrixImpl;
  primitiveMatrixUtil->dot = dotPrimMatrixImpl;
  primitiveMatrixUtil->inv = invPrimMatrixImpl;
  primitiveMatrixUtil->solve = solvePrimMatrixImpl;
  primitiveMatrixUtil->lstsq = lstsqPrimMatrixImpl;
  primitiveMatrixUtil->convolve = convolvePrimMatrixImpl;
//_TODO: cross

  return primitiveMatrixUtil;
}
