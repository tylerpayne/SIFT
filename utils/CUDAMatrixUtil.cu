#include <utils//MatrixUtil.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "kernels/MathKernels.cu"

#ifdef __cplusplus
  extern "C" {
#endif

#ifndef THREADS_PER_BLOCK
  #define THREADS_PER_BLOCK 1024;
#endif

cudaStream_t _stream;
cublasHandle_t _cublasHandle;
cusolverDnHandle_t _cusolverHandle;

DLLEXPORT void cudaSafeCall(cudaError_t stat)
{
  if (stat != cudaSuccess)
  {
    printf("CUDA ERR\n%s\n",cudaGetErrorString(stat));
  }
}

void cublasSafeCall(cublasStatus_t stat)
{
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf("\nCUBLAS ERR: %i\n",stat);
  }
}

void cusolverSafeCall(cusolverStatus_t stat)
{
  if (stat != CUSOLVER_STATUS_SUCCESS)
  {
    printf("\nCUSOLVER ERR: %i\n",stat);
  }
}

void GPUWARN(char* s)
{
    printf("\n\n##### GPU WARNING #####\n");
    printf("%s",s);
    printf("\n###################\n\n");
}

void freeCudaMatrixDeviceMemory(Matrix* A)
{
  cudaSafeCall(cudaFree(A->devicePtr));
}

void freeCudaMatrixImpl(Matrix* A)
{
  if (A->devicePtr != NULL)
  {
    freeCudaMatrixDeviceMemory(A);
  }
  free(A->hostPtr);
  free(A);
}

DLLEXPORT void copyDeviceToDeviceCudaMatrix(Matrix* A, Matrix* B)
{
    size_t size = sizeof(float)*A->shape.width*A->shape.height;
    cudaSafeCall(cudaMemcpy(B->devicePtr,
                            A->devicePtr,
                            size,
                            cudaMemcpyDeviceToDevice));
}

void copyHostToDeviceCudaMatrix(Matrix* A)
{
    if (VERBOSITY > 3)
    {
      GPUWARN("Copying from Host to Device");
    }

    size_t size = sizeof(float)*A->shape.width*A->shape.height;
    cudaSafeCall(cudaMemcpy(A->devicePtr,
                            A->hostPtr,
                            size,
                            cudaMemcpyHostToDevice));
    A->isHostSide = FALSE;
}

void copyDeviceToHostCudaMatrix(Matrix* A)
{
  if (VERBOSITY > 3)
  {
    GPUWARN("Copying Device to Host");
  }

  size_t size = sizeof(float)*A->shape.width*A->shape.height;
  cudaSafeCall(cudaMemcpy(A->hostPtr,
                          A->devicePtr,
                          size,
                          cudaMemcpyDeviceToHost));
  A->isHostSide = TRUE;
}

float getCudaMatrixElementImpl(Matrix* self, Point2 index)
{
  if (self->isHostSide)
  {
    return self->hostPtr[IDX2C(index,self->shape)];
  } else
  {
    return self->devicePtr[IDX2C(index,self->shape)];
  }
}

float* getCudaMatrixRegionImpl(Matrix* self, Rect rect)
{
  float* data;
  size_t s = sizeof(float)*rect.shape.width*rect.shape.height;
  if (self->isHostSide)
  {
    data = (float*)malloc(s);
  }
  else
  {
    cudaSafeCall(cudaMalloc(&data,s));
  }

  for (int y = 0; y < rect.shape.height; y++)
  {
    for (int x = 0; x < rect.shape.width; x++)
    {
      Point2 tmp = {rect.origin.x,rect.origin.y};
      tmp.y += y;
      tmp.x += x;
      data[IDX2C(tmp,rect.shape)] = self->getElement(self,tmp);
    }
  }
  return data;
}

void setCudaMatrixElementImpl(Matrix* self, Point2 index, float x)
{
  if (self->isHostSide)
  {
    self->hostPtr[IDX2C(index,self->shape)] = x;
  }
  else
  {
    self->devicePtr[IDX2C(index,self->shape)] = x;
  }
}

void setCudaMatrixRegionImpl(Matrix* self, Rect rect, float* data)
{
  int counter = 0;
  for (int i = rect.origin.y; i< rect.origin.y+rect.shape.height; i++)
  {
    for (int j = rect.origin.x; j < rect.origin.x+rect.shape.width; j++)
    {
      Point2 idx = {j,i};
      self->setElement(self,idx,data[counter]);
      counter++;
    }
  }
}

Matrix* newEmptyCudaMatrixImpl(Shape shape)
{
  Matrix* m = (Matrix*)malloc(sizeof(Matrix));
  float* h_data = (float*)malloc(sizeof(float)*shape.height*shape.width);
  memset(h_data,0,sizeof(float)*shape.height*shape.width);
  float* d_data;
  cudaSafeCall(cudaMalloc(&d_data,sizeof(float)*shape.height*shape.width));

  m->shape = shape;
  m->hostPtr = h_data;
  m->devicePtr = d_data;
  m->isHostSide = 1;
  m->T = CUBLAS_OP_N;
  m->getElement = getCudaMatrixElementImpl;
  m->getRegion = getCudaMatrixRegionImpl;
  m->setElement = setCudaMatrixElementImpl;
  m->setRegion = setCudaMatrixRegionImpl;
  m->free = freeCudaMatrixImpl;
  m->toDevice = copyHostToDeviceCudaMatrix;
  m->toHost = copyDeviceToHostCudaMatrix;

  return m;
}

Matrix* newCudaMatrixImpl(float* data, Shape shape)
{
  Matrix* m = newEmptyCudaMatrixImpl(shape);
  free(m->hostPtr);
  m->hostPtr = data;
  return m;
}

//################################################
//################################################
//############## BEGIN MATH FUNCS ################
//################################################
//################################################

//ADDITION
void addCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);
  assert(!C->isHostSide);

  copyDeviceToDeviceCudaMatrix(B,C);

  float a = 1;
  cublasSafeCall(cublasSaxpy(_cublasHandle,
                              A->shape.height*A->shape.width,
                              &a,
                              A->devicePtr,1,
                              C->devicePtr,1));
}

void addfCudaMatrixImpl(MatrixUtil* self, Matrix* A, float* b, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!C->isHostSide);

  cublasSafeCall(cublasSaxpy(_cublasHandle,
                              A->shape.height*A->shape.width,
                              &b,
                              A->devicePtr,1,
                              C->devicePtr,1));
}

//SUBTRACTION
void subtractCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);
  assert(!C->isHostSide);

  copyDeviceToDeviceCudaMatrix(A,C);

  float a = -1;
  cublasSafeCall(cublasSaxpy(_cublasHandle,
                              A->shape.height*A->shape.width,
                              &a,
                              B->devicePtr,1,
                              C->devicePtr,1));
}

void subtracftCudaMatrixImpl(MatrixUtil* self, Matrix* A, float b, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!C->isHostSide);

  copyDeviceToDeviceCudaMatrix(A,C);

  (*b) = -1*(*b);
  cublasSafeCall(cublasSaxpy(_cublasHandle,
                              A->shape.height*A->shape.width,
                              &b,
                              A->devicePtr,1,
                              C->devicePtr,1));
}

//MULTIPLY C=A*B
void multiplyCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);
  assert(!C->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  MultiplyElemWiseKernel<<<gdim, bdim,0,_stream>>>(A->devicePtr,B->devicePtr,C->devicePtr,A->shape);
}

void multiplyfCudaMatrixImpl(MatrixUtil* self, Matrix* A, float b, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!C->isHostSide);

  cublasSafeCall(cublasSaxpy(_cublasHandle,
                              A->shape.height*A->shape.width,
                              &b,
                              A->devicePtr,1,
                              C->devicePtr,1));
}


//DIVIDE C=A/B
void divideCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);
  assert(!C->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  DivideElemWiseKernel<<<gdim, bdim,0,_stream>>>(A->devicePtr,B->devicePtr,C->devicePtr,A->shape);
}

void dividefCudaMatrixImpl(MatrixUtil* self, Matrix* A, float b, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!C->isHostSide);

  (*b) = 1.0/(*b);
  cublasSafeCall(cublasSaxpy(_cublasHandle,
                              A->shape.height*A->shape.width,
                              &b,
                              A->devicePtr,1,
                              C->devicePtr,1));
}

//DOT
void dotCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);
  assert(!C->isHostSide);

  float alpha = 1;
  float beta = 0;
  int lda, tda, tdb;
  cublasOperation_t opA, opB;

  if (A->T)
  {
    opA = CUBLAS_OP_T;
    lda = A->shape.width;
    tda = A->shape.height;
  } else
  {
    opA = CUBLAS_OP_N;
    lda = A->shape.height;
    tda = A->shape.width;
  }

  if (B->T)
  {
    opB = CUBLAS_OP_T;
    tdb = B->shape.height;
  } else
  {
    opB = CUBLAS_OP_N;
    tdb = B->shape.width;
  }

  cublasSafeCall(cublasSgemm(_cublasHandle,
                           opA, opB,
                           lda, tdb, tda,
                           &alpha,
                           A->devicePtr, A->shape.height,
                           B->devicePtr, B->shape.height,
                           &beta,
                           C->devicePtr, C->shape.height));
}

void makeCrossMatrix(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);
  Cross3X3MatrixKernel<<<1,1,0,_stream>>>(A->devicePtr,B->devicePtr);
}

//CROSS
void crossCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);
  assert(!C->isHostSide);

  Shape shape = {3,3};
  Matrix* Bx = self->newEmptyMatrix(shape);
  Bx->toDevice(Bx);
  self->makeCrossMatrix(self,B,Bx);
  float alpha = 1;
  float beta = 0;
  cublasSafeCall(cublasSgemv(_cublasHandle,
                                CUBLAS_OP_N, 3, 3,
                                &alpha,
                                Bx->devicePtr, 3,
                                A->devicePtr, 1,
                                &beta,
                                C->devicePtr, 1));
  Bx->free(Bx);
}


/////##############################

void absCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  AbsMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void sqrtCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  SqrtMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void cosCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  CosMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void sinCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  SinMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void tanCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  TanMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void acosCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  ArccosMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void asinCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  ArcsinMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void atanCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  ArctanMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void logCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  LogMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

void expCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int length = A->shape.width*A->shape.height;
  int bdimX = fmin(THREADS_PER_BLOCK,length);
  dim3 bdim(bdimX);
  dim3 gdim(length/bdimX + 1);
  ExpMatrixKernel<<<gdim, bdim, 0, _stream>>>(A->devicePtr,B->devicePtr,A->shape);
}

//############################

//MIN
int* minRowsCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  assert(!A->isHostSide);

  int *idx = (int*)malloc(sizeof(int)*A->shape.height);

  for (int i = 0; i < A->shape.height; i++)
  {
    cublasSafeCall(cublasIsamin(_cublasHandle, A->shape.width,
                            A->devicePtr+(i*A->shape.width), 1, idx+i));
    idx[i] -= 1;
  }
  return idx;
}

float maxCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  int c = self->argmax(self,A);
  Point2 idx = C2IDX(c,A->shape);
  return A->getElement(A,idx);
}

int argmaxCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  assert(!A->isHostSide);

  int idx = 0;
  cublasSafeCall(cublasIsamax(_cublasHandle,
                              A->shape.height*A->shape.width,
                              A->devicePtr, 1,
                              &idx));
  idx -= 1;
  return idx;
}

float minCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  int c = self->argmin(self,A);
  Point2 idx = C2IDX(c,A->shape);
  return A->getElement(A,idx);
}

int argminCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  assert(!A->isHostSide);

  int idx = 0;
  cublasSafeCall(cublasIsamin(_cublasHandle,
                              A->shape.height*A->shape.width,
                              A->devicePtr, 1,
                              &idx));
  idx -= 1;
  return idx;
}

void transposeCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* C)
{
  assert(!A->isHostSide);
  assert(!C->isHostSide);

  int bdimX = fmin(32,A->shape.width);
  int bdimY = fmin(32,A->shape.height);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape.height/bdimX+1,A->shape.width/bdimY + 1);
  TransposeMatrixKernel<<<gdim,bdim,0,_stream>>>(A->devicePtr,C->devicePtr,A->shape.height,A->shape.width);
}
/*
float distanceCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(B);
  }

  Matrix* C = self->newEmptyMatrix(A->shape);

  self->subtract(self,A,B,C);

  float retval;
  cublasSafeCall(cublasSnrm2(_cublasHandle,A->shape.height*A->shape.width,C->devicePtr,1,&retval));
  return retval;
}


void featureDistanceCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
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
  int bdimX = fmin(32,B->shape.height);
  int bdimY = fmin(32,A->shape.height);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(B->shape.height/bdimX + 1,A->shape.height/bdimY + 1);
  FeatureDistanceMatrixKernel<<<gdim,bdim,0,_stream>>>(A->devicePtr,A->shape.height,B->devicePtr,B->shape.height,C->devicePtr,A->shape.width);
}*/

//inv
void invCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  float* Acopy;

  size_t size = sizeof(float)*A->shape.height*A->shape.width;
  cudaSafeCall(cudaMalloc(&Acopy,size));
  cudaSafeCall(cudaMemcpy(Acopy,
                          A->devicePtr,
                          size,
                          cudaMemcpyDeviceToDevice));

  int n = A->shape.height;
  int Lwork;

  //GET BUFFER size
  cusolverSafeCall(cusolverDnSgetrf_bufferSize(_cusolverHandle,
                      n,
                      n,
                      Acopy,
                      n,
                      &Lwork));

  //Create Workspace
  float* workspace;
  cudaSafeCall(cudaMalloc(&workspace,Lwork));

  //Prepare LU decomposition
  int* devIpiv;
  cudaSafeCall(cudaMalloc(&devIpiv,sizeof(int)*n));

  int* devInfo;
  cudaSafeCall(cudaMalloc(&devInfo,sizeof(int)));

  //DECOMPOSE
  cusolverSafeCall(cusolverDnSgetrf(_cusolverHandle,
           n,
           n,
           Acopy,
           n,
           workspace,
           devIpiv,
           devInfo));

  if (VERBOSITY > 3)
  {
    int* h_info = (int*)malloc(sizeof(int));
    cudaSafeCall(cudaMemcpy(h_info,devInfo,sizeof(int),cudaMemcpyDeviceToHost));
    printf("LU DECOMPOSITION INFO: %i\n",h_info[0]);
  }

  //right hand sides
  float B[] = {1,0,0,0,1,0,0,0,1};
  float *d_B;
  cudaSafeCall(cudaMalloc(&d_B,size));
  cudaSafeCall(cudaMemcpy(d_B,&B,size,cudaMemcpyHostToDevice));

  //solve
  cusolverSafeCall(cusolverDnSgetrs(_cusolverHandle,
           CUBLAS_OP_N,
           n,
           n,
           Acopy,
           n,
           devIpiv,
           d_B,
           n,
           devInfo ));

  if (VERBOSITY > 3)
  {
    int* h_info = (int*)malloc(sizeof(int));
    cudaSafeCall(cudaMemcpy(h_info,devInfo,sizeof(int),cudaMemcpyDeviceToHost));
    printf("SOLVE INFO: %i\n",h_info[0]);
  }

  cudaSafeCall(cudaMemcpy(B->devicePtr,d_B,size,cudaMemcpyDeviceToDevice));
  cudaSafeCall(cudaFree(workspace));
  cudaSafeCall(cudaFree(devIpiv));
  cudaSafeCall(cudaFree(devInfo));
  cudaSafeCall(cudaFree(Acopy));
  cudaSafeCall(cudaFree(d_B));
}

void copyCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Rect rect, Point2 Bidx)
{
  assert(!A->isHostSide);
  assert(!B->isHostSide);

  int bdimX = fmin(32,size.shape.width);
  int bdimY = fmin(32,size.shape.height);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(size.shape.width/bdimX + 1,size.shape.height/bdimY + 1);
  CopyMatrixKernel<<<gdim,bdim,0,_stream>>>(A->devicePtr,A->shape.height,A->shape.width,Aidx,B->devicePtr,B->shape.height,B->shape.width,Bidx,size);
}

Matrix* sliceCudaMatrixImpl(MatrixUtil* self, Matrix* A, Rect rect)
{
  assert(!A->isHostSide);

  Matrix* B = self->newEmptyMatrix(self,rect.size);
  B->toDevice(B);
  Point2 Bidx = {0,0};
  self->copy(self,A,B,rec,Bidx);
  return B;
}

DLLEXPORT void SetCUDAMatrixUtilStream(cudaStream_t stream)
{
    _stream = stream;
    cublasSetStream(_cublasHandle, stream);
    cusolverDnSetStream(_cusolverHandle, stream);
}

void InitCUDAHandles()
{
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  _cublasHandle = cublasHandle;

  cusolverDnHandle_t cusolverHandle;
  cusolverDnCreate(&cusolverHandle);
  _cusolverHandle = cusolverHandle;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  _stream = stream;
  SetCUDAMatrixUtilStream(stream);
}

void pprintCudaMatrixImpl(MatrixUtil* self, Matrix* A, char* label)
{
  printf("\n\n############### %s #####################",label);
  if (!A->isHostSide)
  {
    A->toHost(A);
  }
  for (int i = 0; i < A->shape.height; i++)
  {
    for (int j = 0;j< A->shape.width;j++)
    {
      Point2 idx = {j,i};
      printf("[ %f ]",A->getElement(A,idx));
    }
    printf("\n");
  }
  printf("\n################################################\n\n");
}

DLLEXPORT MatrixUtil* GetMatrixUtil()
{
  MatrixUtil* self = (MatrixUtil*)malloc(sizeof(MatrixUtil));
  InitCUDAHandles();

  self->newEmptyMatrix = newEmptyCudaMatrixImpl;
  self->newMatrix = newCudaMatrixImpl;
  self->copy = copyCudaMatrixImpl;
  self->pprint = pprintCudaMatrixImpl;

  self->add = addCudaMatrixImpl;
  self->addf = addfCudaMatrixImpl;
  self->subtract = subtractCudaMatrixImpl;
  self->subtractf = subtractfCudaMatrixxImpl;
  self->multiply = multiplyCudaMatrixImpl;
  self->multiplyf = multiplyfCudaMatrixImpl;
  self->divide = divideCudaMatrixImpl;
  self->dividef = dividefCudaMatrixImpl;
  self->dot = dotCudaMatrixImpl;
  self->cross = cross3X3MatrixImpl;

  self->abs = absCudaMatrixImpl;
  self->sqrt = sqrtCudaMatrixImpl;
  self->cos = cosCudaMatrixImpl;
  self->sin = sinCudaMatrixImpl;
  self->tan = tanCudaMatrixImpl;
  self->acos = acosCudaMatrixImpl;
  self->asin = asinCudaMatrixImpl;
  self->atan = atanCudaMatrixImpl;
  self->exp = expCudaMatrixImpl;
  self->log = logCudaMatrixImpl;

  self->inv = invCudaMatrixImpl;

  self->max = maxCudaMatrixImpl
  self->argmax = argmaxCudaMatrixImpl;
  self->min = minCudaMatrixImpl;
  self->argmin = argminCudaMatrixImpl;

  return self;
}

#ifdef __cplusplus
  }
#endif
