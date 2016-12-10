#include <stdio.h>
#include <math.h>
#include <string.h>
#include "MatrixUtil.h"
#include "MathKernels.cu"
#include "ImageKernels.cu"
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

cudaError_t cudaErr;
cublasStatus_t cublasStat;

void freeCudaMatrixDeviceMemory(Matrix* mat)
{
  cudaFree((float*)mat->devicePtr);
}

void copyDeviceToDeviceCudaMatrix(MatrixUtil* self, Matrix* A, Matrix* B)
{
    cudaSetDevice(self->deviceId);
    size_t size = sizeof(float)*A->shape[0]*A->shape[1];
    cudaMemcpyAsync((float*)B->devicePtr,(float*)A->devicePtr,size,cudaMemcpyDeviceToDevice,self->stream);
    B->isHostSide = 0;
    //cudaMemcpyAsync((float*)mat->devicePtr,(float*)mat->nativePtr,size,cudaMemcpyHostToDevice,self->stream);
}

void copyHostToDeviceCudaMatrix(MatrixUtil* self, Matrix* mat)
{
    cudaSetDevice(self->deviceId);
    size_t size = sizeof(float)*mat->shape[0]*mat->shape[1];
    if (VERBOSITY > 3)
    {
      printf("\n\n##### GPU WARNING #####\n");
      printf("Copying from Host to Device");
      printf("\n###################\n\n");
    }
    //cudaMemcpy((float*)mat->devicePtr,(float*)mat->nativePtr,size,cudaMemcpyHostToDevice);
    cudaMemcpyAsync((float*)mat->devicePtr,(float*)mat->nativePtr,size,cudaMemcpyHostToDevice,self->stream);

    mat->isHostSide = 0;
}

void copyDeviceToHostCudaMatrix(MatrixUtil* self, Matrix* mat)
{
  cudaSetDevice(self->deviceId);
  if (VERBOSITY > 3)
  {
    printf("\n\n### GPU WARNING ###\n");
    printf("Copying Device to Host");
    printf("\n###################\n\n");
  }
  size_t size = sizeof(float)*mat->shape[0]*mat->shape[1];
  //cudaMemcpy((float*)mat->nativePtr,(float*)mat->devicePtr,size,cudaMemcpyDeviceToHost);
  cudaMemcpyAsync((float*)mat->nativePtr,(float*)mat->devicePtr,size,cudaMemcpyDeviceToHost,self->stream);
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
  float* d_data;
  cudaMalloc(&d_data,sizeof(float)*width*height);
  int* shape = (int*)malloc(sizeof(int)*2);
  shape[0] = width;
  shape[1] = height;

  m->shape = shape;
  m->nativePtr = (void*)h_data;
  m->devicePtr = (void*)d_data;
  m->isHostSide = 1;
  m->T = CUBLAS_OP_N;
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
  //copyHostToDeviceCudaMatrix(m);
  return m;
}
//############ BEGIN MATH FUNCS ##################
//ADDITION
void addCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,B);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,C);
  }
  cudaStreamSynchronize(self->stream);
  copyDeviceToDeviceCudaMatrix(self,B,C);
  cudaStreamSynchronize(self->stream);
  cublasSetStream(self->cublasHandle,self->stream);
  float a = 1;
  cublasSaxpy(self->cublasHandle,A->shape[0]*A->shape[1],&a,(float*)A->devicePtr,1,(float*)C->devicePtr,1);
}

//SUBTRACTION
void subtractCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,B);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,C);
  }
  cudaStreamSynchronize(self->stream);
  copyDeviceToDeviceCudaMatrix(self,A,C);
  cudaStreamSynchronize(self->stream);
  cublasSetStream(self->cublasHandle,self->stream);
  float a = -1;
  cublasSaxpy(self->cublasHandle,A->shape[0]*A->shape[1],&a,(float*)B->devicePtr,1,(float*)C->devicePtr,1);
}

//MULTIPLYCONST
void multiplyConstCudaMatrixImpl(MatrixUtil* self, Matrix* A, float b, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,C);
  }
  cudaStreamSynchronize(self->stream);
  cublasSetStream(self->cublasHandle,self->stream);
}

//DOT
void dotCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,B);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(self,C);
  }
  cudaStreamSynchronize(self->stream);
  cublasSetStream(self->cublasHandle,self->stream);
  float alpha = 1;
  float beta = 0;
  cublasSgemm(self->cublasHandle,A->T,B->T,A->shape[0],B->shape[1],A->shape[1],&alpha,(float*)A->devicePtr,A->shape[0],(float*)B->devicePtr,B->shape[0],&beta,(float*)C->devicePtr,C->shape[0]);
}

/*
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
  int bdimX = fmin(32,(A->shape[0]/32 + 1)*32/32 + 1);
  int bdimY = fmin(32,(A->shape[1]/32 + 1)*32/32 + 1);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
  if (VERBOSITY > 2)
  {
    printf("LAUNCHING MULTCONST KERNEL\n BlockDim: (%i,%i) GridDim: (%i,%i)\n\n",bdim.x,bdim.y,gdim.x,gdim.y);
    printf("\nA dim: (%i,%i)\nB: %f\nC dim: (%i,%i)\n\n",A->shape[0],A->shape[1],B,C->shape[0],C->shape[1]);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
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
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
  if (VERBOSITY > 2)
  {
    printf("LAUNCHING CONVOLVE KERNEL\n BlockDim: (%i,%i) GridDim: (%i,%i)\n\n",bdim.x,bdim.y,gdim.x,gdim.y);
    printf("\nA dim: (%i,%i)\nB dim: (%i,%i)\nC dim: (%i,%i)\n\n",A->shape[0],A->shape[1],B->shape[0],B->shape[1],C->shape[0],C->shape[1]);
  }
  MatConvolve<<<gdim,bdim>>>((float*)A->devicePtr,(float*)B->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1],B->shape[0],B->shape[1]);
}
Matrix* transposeCudaMatrixImpl(Matrix* A)
{
  Matrix* C = newEmptyCudaMatrixImpl(A->shape[1],A->shape[0]);
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,(A->shape[0] + 32 - 1) / 32);
  int bdimY = fmin(32,(A->shape[1] + 32 - 1) / 32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim((A->shape[0] + bdimX - 1)/bdimX,(A->shape[1] + bdimY - 1)/bdimY);
  MatTranspose<<<gdim,bdim>>>((float*)A->devicePtr,(float*)C->devicePtr,A->shape[0],A->shape[1]);
  return C;
}

//DOT
void dotCudaMatrixImpl(Matrix* A, Matrix* B, Matrix* C)
{
  B = transposeCudaMatrixImpl(B);
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

  int bdimX = fmin(32,(C->shape[0]/32 + 1)*32);
  int bdimY = fmin(32,(C->shape[1]/32 + 1)*32);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(((C->shape[0]/bdimX)/32 + 1)*32,((C->shape[1]/bdimY)/32 + 1)*32);
  if (VERBOSITY > 2)
  {
    printf("LAUNCHING DOT KERNEL\n BlockDim: (%i,%i) GridDim: (%i,%i)\n\n",bdim.x,bdim.y,gdim.x,gdim.y);
    printf("\nA dim: (%i,%i)\nB dim: (%i,%i)\nC dim: (%i,%i)\n\n",A->shape[0],A->shape[1],B->shape[0],B->shape[1],C->shape[0],C->shape[1]);
  }
  MatDot<<<gdim,bdim>>>((float*)A->devicePtr,(float*)B->devicePtr,(float*)C->devicePtr,A->shape[0],B->shape[1],B->shape[0]);
}

//##################### END MATH FUNCS ####################

//##################### TEMP Cuda LA FUNCS ##############
Matrix* invCudaMatrixImpl(Matrix* A)
{
  if (!A->isHostSide)
  {
    copyDeviceToHostCudaMatrix(A);
  }
  int height = A->shape[0];
  int width = A->shape[1];
  float* solverData = (float*)malloc(sizeof(float)*height*width);
  memcpy(solverData,A->nativePtr,sizeof(float)*height*width);
  float* resultsData = (float*)malloc(sizeof(float)*height*width);
  for (int j = 0; j < width; j++)
  {
    resultsData[IDX2C(j,j,width)] = 1.0;
  }
  Matrix* solver = newCudaMatrixImpl(solverData,height,width);
  Matrix* results = newCudaMatrixImpl(resultsData,height,width);
  for (int col = 0; col<width-1; col++)
  {
    Matrix* pivot = newCudaMatrixImpl(solver->getRegion(solver,col,0,1,width),1,width);
    float normalize = pivot->getElement(pivot,0,col);
    divideConstCudaMatrixImpl(pivot,normalize,pivot);
    solver->setRegion(solver,col,0,1,width,pivot->getRegion(pivot,0,0,1,width));
    Matrix* resultPivot = newCudaMatrixImpl(results->getRegion(results,col,0,1,width),1,width);
    divideConstCudaMatrixImpl(resultPivot,normalize,resultPivot);
    results->setRegion(results,col,0,1,width,resultPivot->getRegion(resultPivot,0,0,1,width));
    for (int row = col+1; row<height; row++)
    {
      Matrix *thisRow = newCudaMatrixImpl(solver->getRegion(solver,row,0,1,width),1,width);
      Matrix* thisResultRow = newCudaMatrixImpl(results->getRegion(results,row,0,1,width),1,width);
      float ratio = thisRow->getElement(thisRow,0,col) / pivot->getElement(pivot,0,col);
      Matrix* scaledPivot = newEmptyCudaMatrixImpl(pivot->shape[0],pivot->shape[1]);
      multiplyConstCudaMatrixImpl(pivot,ratio,scaledPivot);
      Matrix* updatedRow = newEmptyCudaMatrixImpl(thisRow->shape[0],thisRow->shape[1]);
      subtractCudaMatrixImpl(thisRow,scaledPivot,updatedRow);
      Matrix* scaledResult = newEmptyCudaMatrixImpl(resultPivot->shape[0],resultPivot->shape[1]);
      multiplyConstCudaMatrixImpl(resultPivot,ratio,scaledResult);
      Matrix* updatedResult = newEmptyCudaMatrixImpl(thisResultRow->shape[0],thisResultRow->shape[1]);
      subtractCudaMatrixImpl(thisResultRow,scaledResult,updatedResult);
      //pprintCudaMatrixImpl(updatedRow,"updatedRow");
      //pprintCudaMatrixImpl(thisRow,"thisRow");
      results->setRegion(results,row,0,1,width,updatedResult->getRegion(updatedResult,0,0,1,width));
      solver->setRegion(solver,row,0,1,width,updatedRow->getRegion(updatedRow,0,0,1,width));
    }
  }
  //Norm bottom right element
  Matrix* pivot = newCudaMatrixImpl(solver->getRegion(solver,width-1,0,1,width),1,width);
  float normalize = pivot->getElement(pivot,0,width-1);
  divideConstCudaMatrixImpl(pivot,normalize,pivot);
  solver->setRegion(solver,width-1,0,1,width,pivot->getRegion(pivot,0,0,1,width));
  Matrix* resultPivot = newCudaMatrixImpl(results->getRegion(results,width-1,0,1,width),1,width);
  divideConstCudaMatrixImpl(resultPivot,normalize,resultPivot);
  results->setRegion(results,width-1,0,1,width,resultPivot->getRegion(resultPivot,0,0,1,width));

  for (int col = width-1; col>=0; col--)
  {
    Matrix* pivot = newCudaMatrixImpl(solver->getRegion(solver,col,0,1,width),1,width);
    Matrix* resultPivot = newCudaMatrixImpl(results->getRegion(results,col,0,1,width),1,width);
    for (int row = col-1; row>=0; row--)
    {
      Matrix *thisRow = newCudaMatrixImpl(solver->getRegion(solver,row,0,1,width),1,width);
      Matrix* thisResultRow = newCudaMatrixImpl(results->getRegion(results,row,0,1,width),1,width);
      float ratio = thisRow->getElement(thisRow,0,col) / pivot->getElement(pivot,0,col);
      Matrix* scaledPivot = newEmptyCudaMatrixImpl(pivot->shape[0],pivot->shape[1]);
      multiplyConstCudaMatrixImpl(pivot,ratio,scaledPivot);
      Matrix* updatedRow = newEmptyCudaMatrixImpl(thisRow->shape[0],thisRow->shape[1]);
      subtractCudaMatrixImpl(thisRow,scaledPivot,updatedRow);
      Matrix* scaledResult = newEmptyCudaMatrixImpl(resultPivot->shape[0],resultPivot->shape[1]);
      multiplyConstCudaMatrixImpl(resultPivot,ratio,scaledResult);
      Matrix* updatedResult = newEmptyCudaMatrixImpl(thisResultRow->shape[0],thisResultRow->shape[1]);
      subtractCudaMatrixImpl(thisResultRow,scaledResult,updatedResult);
      results->setRegion(results,row,0,1,width,updatedResult->getRegion(updatedResult,0,0,1,width));
      solver->setRegion(solver,row,0,1,width,updatedRow->getRegion(updatedRow,0,0,1,width));
    }
  }
  for (int col = width-1; col>=0; col--)
  {
    Matrix* pivot = newCudaMatrixImpl(solver->getRegion(solver,col,0,1,width),1,width);
    Matrix* resultPivot = newCudaMatrixImpl(results->getRegion(results,col,0,1,width),1,width);
    float divisor = pivot->getElement(pivot,0,col);
    divideConstCudaMatrixImpl(pivot,divisor,pivot);
    divideConstCudaMatrixImpl(resultPivot,divisor,resultPivot);
    results->setRegion(results,col,0,1,width,resultPivot->getRegion(resultPivot,0,0,1,width));
    solver->setRegion(solver,col,0,1,width,pivot->getRegion(pivot,0,0,1,width));
  }
  return results;
}

Matrix* solveCudaMatrixImpl(Matrix* A, Matrix* y)
{
  if (!A->isHostSide)
  {
    copyDeviceToHostCudaMatrix(A);
  }
  if (!y->isHostSide)
  {
    copyDeviceToHostCudaMatrix(y);
  }
  int height = A->shape[0];
  int width = A->shape[1];
  float* solverData = (float*)malloc(sizeof(float)*height*width);
  memcpy(solverData,A->nativePtr,sizeof(float)*height*width);
  float* resultsData = (float*)malloc(sizeof(float)*height);
  memcpy(resultsData,y->nativePtr,sizeof(float)*height);
  Matrix* solver = newCudaMatrixImpl(solverData,height,width);
  Matrix* results = newCudaMatrixImpl(resultsData,height,1);

  for (int col = 0; col<width-1; col++)
  {
    Matrix* pivot = newCudaMatrixImpl(solver->getRegion(solver,col,0,1,width),1,width);
    float resultPivot = results->getElement(results,col,0);
    for (int row = col+1; row<height; row++)
    {
      Matrix *thisRow = newCudaMatrixImpl(solver->getRegion(solver,row,0,1,width),1,width);
      float ratio = thisRow->getElement(thisRow,0,col) / pivot->getElement(pivot,0,col);
      Matrix* scaledPivot = newEmptyCudaMatrixImpl(pivot->shape[0],pivot->shape[1]);
      multiplyConstCudaMatrixImpl(pivot,ratio,scaledPivot);
      Matrix* updatedRow = newEmptyCudaMatrixImpl(thisRow->shape[0],thisRow->shape[1]);
      subtractCudaMatrixImpl(thisRow,scaledPivot,updatedRow);
      float scaledResult = resultPivot*ratio;
      float updatedResult = results->getElement(results,row,0) - scaledResult;
      results->setElement(results,row,0,updatedResult);
      solver->setRegion(solver,row,0,1,width,updatedRow->getRegion(updatedRow,0,0,1,width));
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
*/

void pprintCudaMatrixImpl(MatrixUtil* self, Matrix* A, char* label)
{
  if (!A->isHostSide)
  {
    printf("copying to host");
    copyDeviceToHostCudaMatrix(self,A);
  }
  cudaStreamSynchronize(self->stream);
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

/*
Matrix* lstsqCudaMatrixImpl(Matrix* A, Matrix* b)
{
  if (!A->isHostSide)
  {
    copyDeviceToHostCudaMatrix(A);
  }
  if (!b->isHostSide)
  {
    copyDeviceToHostCudaMatrix(b);
  }
  int observations = A->shape[0];
  int parameters = A->shape[1];
  float* XData = (float*)malloc(sizeof(float)*observations*parameters);
  memcpy(XData,A->nativePtr,sizeof(float)*observations*parameters);
  float* yData = (float*)malloc(sizeof(float)*observations);
  memcpy(yData,b->nativePtr,sizeof(float)*observations);
  Matrix* X = newCudaMatrixImpl(XData,observations,parameters);
  Matrix* y = newCudaMatrixImpl(yData,observations,1);
  Matrix* XT = transposeCudaMatrixImpl(X);
  pprintCudaMatrixImpl(X,"X");
  pprintCudaMatrixImpl(y,"Y");
  pprintCudaMatrixImpl(XT,"XT");
  Matrix* Gramian = newEmptyCudaMatrixImpl(XT->shape[0],X->shape[1]);
  dotCudaMatrixImpl(XT,X,Gramian);
  pprintCudaMatrixImpl(Gramian,"Gramian");
  Gramian = invCudaMatrixImpl(Gramian);
  pprintCudaMatrixImpl(Gramian,"invGramian");
  Matrix* XTy = newEmptyCudaMatrixImpl(XT->shape[0],y->shape[1]);
  dotCudaMatrixImpl(XT,y,XTy);
  pprintCudaMatrixImpl(XTy,"XTy");
  Matrix* retval = newEmptyCudaMatrixImpl(Gramian->shape[0],XTy->shape[1]);
  dotCudaMatrixImpl(Gramian,XTy,retval);
  return retval;
}

//##################### END Cuda LA FUNCS ################

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


void syncCudaMatrixImpl(Matrix* m)
{
  if (m->isHostSide)
  {
    copyHostToDeviceCudaMatrix(m);
  } else
  {
    copyDeviceToHostCudaMatrix(m);
  }
}*/

void SetCUDAMatrixUtilStream(MatrixUtil* self, cudaStream_t stream)
{
    self->stream = stream;
}

void SetCUDAMatrixUtilDevice(MatrixUtil* self, int device)
{
  self->deviceId = device;
  cudaSetDevice(device);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  SetCUDAMatrixUtilStream(self,stream);

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  self->cublasHandle = cublasHandle;
}

MatrixUtil* GetCUDAMatrixUtil(int device)
{
  MatrixUtil* cudaMatrixUtil = (MatrixUtil*)malloc(sizeof(MatrixUtil));
  SetCUDAMatrixUtilDevice(cudaMatrixUtil,device);

  cudaMatrixUtil->newEmptyMatrix = newEmptyCudaMatrixImpl;
  cudaMatrixUtil->newMatrix = newCudaMatrixImpl;
  cudaMatrixUtil->add = addCudaMatrixImpl;
  cudaMatrixUtil->subtract = subtractCudaMatrixImpl;
  cudaMatrixUtil->pprint = pprintCudaMatrixImpl;
  cudaMatrixUtil->dot = dotCudaMatrixImpl;
  cudaMatrixUtil->multiplyConst = multiplyConstCudaMatrixImpl;

/*  cudaMatrixUtil->downsample = downsampleCudaMatrixImpl;
  cudaMatrixUtil->sync = syncCudaMatrixImpl;
  cudaMatrixUtil->multiply = multiplyCudaMatrixImpl;
  cudaMatrixUtil->divide = divideCudaMatrixImpl;
  cudaMatrixUtil->divideConst = divideConstCudaMatrixImpl;
  cudaMatrixUtil->pow = powCudaMatrixImpl;
  cudaMatrixUtil->convolve = convolveCudaMatrixImpl;
  cudaMatrixUtil->sqrt = sqrtCudaMatrixImpl;
  //cudaMatrixUtil->isEqual = isEqualCudaMatrixImpl;
  cudaMatrixUtil->arctan = arctanCudaMatrixImpl;
  cudaMatrixUtil->transpose = transposeCudaMatrixImpl;

  cudaMatrixUtil->inv = invCudaMatrixImpl;
  cudaMatrixUtil->solve = solveCudaMatrixImpl;
  cudaMatrixUtil->lstsq = lstsqCudaMatrixImpl;*/
  //cudaMatrixUtil->exp = expCudaMatrixImpl;
  //cudaMatrixUtil->log = logCudaMatrixImpl;

  //cudaMatrixUtil->ceil = ceilCudaMatrixImpl;
  //cudaMatrixUtil->floor = floorCudaMatrixImpl;
  //cudaMatrixUtil->abs = absCudaMatrixImpl;
  /*
  cudaMatrixUtil->convolve = convolveCudaMatrixImpl;*/
//_TODO: cross

  return cudaMatrixUtil;
}
