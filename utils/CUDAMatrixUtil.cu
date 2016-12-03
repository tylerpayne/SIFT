#include <stdio.h>
#include <math.h>
#include <string.h>
#include "MatrixUtil.h"
#include "MathKernels.cu"
#include "ImageKernels.cu"

void freeCudaMatrixDeviceMemory(Matrix* mat)
{
  cudaFree((float*)mat->devicePtr);
}

void copyHostToDeviceCudaMatrix(Matrix* mat)
{
    size_t size = sizeof(float)*mat->shape[0]*mat->shape[1];
    if (mat->devicePtr == NULL)
    {
      if (VERBOSITY > 3)
      {
        printf("\n\n### GPU WARNING ###\n");
        printf("Matrix devicePtr was empty.\nAllocating %i bytes on the GPU",size);
        printf("\n###################\n\n");
      }
      float* d_data;
      cudaMalloc(&d_data,size);
      cudaMemcpy(d_data,(float*)mat->nativePtr,size,cudaMemcpyHostToDevice);
      mat->devicePtr = (void*)d_data;
    }
    cudaMemcpy((float*)mat->devicePtr,(float*)mat->nativePtr,size,cudaMemcpyHostToDevice);
    mat->isHostSide = 0;
}

void copyDeviceToHostCudaMatrix(Matrix* mat)
{
  if (VERBOSITY > 3)
  {
    printf("\n\n### GPU WARNING ###\n");
    printf("Syncing");
    printf("\n###################\n\n");
  }
  size_t size = sizeof(float)*mat->shape[0]*mat->shape[1];
  cudaMemcpy((float*)mat->nativePtr,(float*)mat->devicePtr,size,cudaMemcpyDeviceToHost);
  mat->isHostSide = 1;
}

float getCudaMatrixElementImpl(Matrix* self, int i, int  j)
{
  if (!self->isHostSide)
  {
    if (VERBOSITY > 3)
    {
      printf("\n### GPU WARNING ###\n");
      printf("Matrix was on device when trying to get.");
      printf("\n###################\n");
    }
    copyDeviceToHostCudaMatrix(self);
  }
  return ((float*)self->nativePtr)[IDX2C(i,j,self->shape[1])];
}

float* getCudaMatrixRegionImpl(Matrix* self, int i, int j, int h, int w)
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

void setCudaMatrixElementImpl(Matrix* self, int i, int  j, float x)
{
  if (!self->isHostSide)
  {
    if (VERBOSITY > 3)
    {
      printf("\n### GPU WARNING ###\n");
      printf("Matrix was on device when trying to set!\n");
      printf("\n###################\n");
    }
    copyDeviceToHostCudaMatrix(self);
  }
  ((float*)self->nativePtr)[IDX2C(i,j,self->shape[1])] = x;
}

void setCudaMatrixRegionImpl(Matrix* self, int i, int j, int r, int c, float* data)
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

Matrix* newEmptyCudaMatrixImpl(int width, int height)
{
  Matrix* m = (Matrix*)malloc(sizeof(Matrix));
  float* h_data = (float*)malloc(sizeof(float)*width*height);
  for (int i = 0; i < width*height; i++)
  {
    h_data[i] = 0.0;
  }
  int* shape = (int*)malloc(sizeof(int)*2);
  shape[0] = width;
  shape[1] = height;

  m->shape = shape;
  m->nativePtr = (void*)h_data;
  m->devicePtr = NULL;
  m->isHostSide = 1;
  m->getElement = getCudaMatrixElementImpl;
  m->getRegion = getCudaMatrixRegionImpl;
  m->setElement = setCudaMatrixElementImpl;
  m->setRegion = setCudaMatrixRegionImpl;
  return m;
}

Matrix* newCudaMatrixImpl(float* data, int width, int height)
{
  Matrix* m = newEmptyCudaMatrixImpl(width,height);
  free(m->nativePtr);
  m->nativePtr = (void*)data;
  return m;
}
//############ BEGIN MATH FUNCS ##################
//ADDITION
void addCudaMatrixImpl(Matrix* A, Matrix* B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(B);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatAdd<<<gdim,bdim>>>((float*)A->devicePtr,(float*)B->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

void addConstCudaMatrixImpl(Matrix* A, float B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatAddConst<<<gdim,bdim>>>((float*)A->devicePtr,B,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}
//SUBTRACTION
void subtractCudaMatrixImpl(Matrix* A, Matrix* B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(B);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  if (VERBOSITY > 2)
  {
    printf("LAUNCHING SUBTRACT KERNEL\n BlockDim: (%i,%i) GridDim: (%i,%i)\n\n",bdim.x,bdim.y,gdim.x,gdim.y);
    printf("\nA dim: (%i,%i)\nC dim: (%i,%i)\n\n",A->shape[0],A->shape[1],C->shape[0],C->shape[1]);
  }
  MatSubtract<<<gdim,bdim>>>((float*)A->devicePtr,(float*)B->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

void subtractConstCudaMatrixImpl(Matrix* A, float B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatSubtractConst<<<gdim,bdim>>>((float*)A->devicePtr,B,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

//MULTIPLICATION

void multiplyCudaMatrixImpl(Matrix* A, Matrix* B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(B);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatMult<<<gdim,bdim>>>((float*)A->devicePtr,(float*)B->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

void multiplyConstCudaMatrixImpl(Matrix* A, float B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  if (VERBOSITY > 2)
  {
    printf("LAUNCHING MULTCONST KERNEL\n BlockDim: (%i,%i) GridDim: (%i,%i)\n\n",bdim.x,bdim.y,gdim.x,gdim.y);
    printf("\nA dim: (%i,%i)\nC dim: (%i,%i)\n\n",A->shape[0],A->shape[1],C->shape[0],C->shape[1]);
  }
  MatMultConst<<<gdim,bdim>>>((float*)A->devicePtr,B,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

//Division
void divideCudaMatrixImpl(Matrix* A, Matrix* B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(B);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatDivide<<<gdim,bdim>>>((float*)A->devicePtr,(float*)B->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

void divideConstCudaMatrixImpl(Matrix* A, float B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatDivideConst<<<gdim,bdim>>>((float*)A->devicePtr,B,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

void powCudaMatrixImpl(Matrix* A, float B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatPow<<<gdim,bdim>>>((float*)A->devicePtr,B,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

void sqrtCudaMatrixImpl(Matrix* A, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatSqrt<<<gdim,bdim>>>((float*)A->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}

void arctanCudaMatrixImpl(Matrix* A, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  MatArctan<<<gdim,bdim>>>((float*)A->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1]);
}


//CONVOLVE
void convolveCudaMatrixImpl(Matrix* A, Matrix* B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(B);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[0]);
  int bdimY = fmin(32,A->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX + 1,A->shape[1]/bdimY + 1);
  if (VERBOSITY > 2)
  {
    printf("LAUNCHING CONVOLVE KERNEL\n BlockDim: (%i,%i) GridDim: (%i,%i)\n\n",bdim.x,bdim.y,gdim.x,gdim.y);
    printf("\nA dim: (%i,%i)\nB dim: (%i,%i)\nC dim: (%i,%i)\n\n",A->shape[0],A->shape[1],B->shape[0],B->shape[1],C->shape[0],C->shape[1]);
  }
  MatConvolve<<<gdim,bdim>>>((float*)A->devicePtr,(float*)B->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1],B->shape[0],B->shape[1]);
}
//##################### END MATH FUNCS ####################

void downsampleCudaMatrixImpl(Matrix* A, Matrix* B)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(B);
  }
  int bdimX = fmin(32,B->shape[0]);
  int bdimY = fmin(32,B->shape[1]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(B->shape[0]/bdimX + 1,B->shape[1]/bdimY + 1);
  if (VERBOSITY > 2)
  {
    printf("LAUNCHING DOWNSAMPLE KERNEL\n BlockDim: (%i,%i) GridDim: (%i,%i)\n\n",bdim.x,bdim.y,gdim.x,gdim.y);
    printf("\nA dim: (%i,%i)\nB dim: (%i,%i)\n\n",A->shape[0],A->shape[1],B->shape[0],B->shape[1]);
  }
  DownsampleKernel<<<gdim,bdim>>>((float*)A->devicePtr,(float*)B->devicePtr,A->shape[0],A->shape[1],B->shape[0],B->shape[1]);
}

void pprintCudaMatrixImpl(Matrix* A, char* label)
{
  if (!A->isHostSide)
  {
    copyDeviceToHostCudaMatrix(A);
  }
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

void syncCudaMatrixImpl(Matrix* m)
{
  if (m->isHostSide)
  {
    copyHostToDeviceCudaMatrix(m);
  } else
  {
    copyDeviceToHostCudaMatrix(m);
  }
}

MatrixUtil* GetCUDAMatrixUtil()
{
  MatrixUtil* cudaMatrixUtil = (MatrixUtil*)malloc(sizeof(MatrixUtil));
  cudaMatrixUtil->newEmptyMatrix = newEmptyCudaMatrixImpl;
  cudaMatrixUtil->newMatrix = newCudaMatrixImpl;
  cudaMatrixUtil->downsample = downsampleCudaMatrixImpl;
  cudaMatrixUtil->sync = syncCudaMatrixImpl;
  cudaMatrixUtil->pprint = pprintCudaMatrixImpl;
  cudaMatrixUtil->add = addCudaMatrixImpl;
  cudaMatrixUtil->subtract = subtractCudaMatrixImpl;
  cudaMatrixUtil->multiply = multiplyCudaMatrixImpl;
  cudaMatrixUtil->multiplyConst = multiplyConstCudaMatrixImpl;
  cudaMatrixUtil->divide = divideCudaMatrixImpl;
  cudaMatrixUtil->divideConst = divideConstCudaMatrixImpl;
  cudaMatrixUtil->pow = powCudaMatrixImpl;
  cudaMatrixUtil->sync = syncCudaMatrixImpl;
  cudaMatrixUtil->convolve = convolveCudaMatrixImpl;
  cudaMatrixUtil->sqrt = sqrtCudaMatrixImpl;
  //cudaMatrixUtil->isEqual = isEqualCudaMatrixImpl;
  cudaMatrixUtil->arctan = arctanCudaMatrixImpl;
  //cudaMatrixUtil->exp = expCudaMatrixImpl;
  //cudaMatrixUtil->log = logCudaMatrixImpl;

  //cudaMatrixUtil->ceil = ceilCudaMatrixImpl;
  //cudaMatrixUtil->floor = floorCudaMatrixImpl;
  //cudaMatrixUtil->abs = absCudaMatrixImpl;
  /*
  cudaMatrixUtil->transpose = transposeCudaMatrixImpl;
  cudaMatrixUtil->dot = dotCudaMatrixImpl;
  cudaMatrixUtil->inv = invCudaMatrixImpl;
  cudaMatrixUtil->solve = solveCudaMatrixImpl;
  cudaMatrixUtil->lstsq = lstsqCudaMatrixImpl;
  cudaMatrixUtil->convolve = convolveCudaMatrixImpl;*/
//_TODO: cross

  return cudaMatrixUtil;
}
