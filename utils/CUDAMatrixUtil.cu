#include "MatrixUtil.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "kernels/MathKernels.cu"

#ifdef __cplusplus
  extern "C" {
#endif

cudaStream_t _stream;
cublasHandle_t _cublasHandle;
cusolverDnHandle_t _cusolverHandle;

void cudaErrCheck(cudaError_t stat)
{
  if (stat != cudaSuccess)
  {
    printf("CUDA ERR: %i\n",stat);
  }
}

void cublasErrCheck(cublasStatus_t stat)
{
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf("\nCUBLAS ERR: %i\n",stat);
  }
}

void cusolverErrCheck(cusolverStatus_t stat)
{
  if (stat != CUSOLVER_STATUS_SUCCESS)
  {
    printf("\nCUSOLVER ERR: %i\n",stat);
  }
}

void freeCudaMatrixDeviceMemory(Matrix* mat)
{
  cudaErrCheck(cudaFree(mat->devicePtr));
}

void freeCudaMatrixImpl(Matrix* m)
{
  freeCudaMatrixDeviceMemory(m);
  free(m->shape);
  free(m->hostPtr);
  free(m);
}

DLLEXPORT void copyDeviceToDeviceCudaMatrix(Matrix* A, Matrix* B)
{
    size_t size = sizeof(float)*A->shape[0]*A->shape[1];
    cudaErrCheck(cudaMemcpy(B->devicePtr,A->devicePtr,size,cudaMemcpyDeviceToDevice));
}

DLLEXPORT void copyHostToDeviceCudaMatrix(Matrix* mat)
{
    size_t size = sizeof(float)*mat->shape[0]*mat->shape[1];
    if (VERBOSITY > 3)
    {
      printf("\n\n##### GPU WARNING #####\n");
      printf("Copying from Host to Device");
      printf("\n###################\n\n");
    }
    cudaErrCheck(cudaMemcpy(mat->devicePtr,mat->hostPtr,size,cudaMemcpyHostToDevice));
    mat->isHostSide = 0;
}

DLLEXPORT void copyDeviceToHostCudaMatrix(Matrix* mat)
{
  size_t size = sizeof(float)*mat->shape[0]*mat->shape[1];
  if (VERBOSITY > 3)
  {
    printf("\n\n### GPU WARNING ###\n");
    printf("Copying Device to Host");
    printf("\n###################\n\n");
  }
  cudaErrCheck(cudaMemcpy(mat->hostPtr,mat->devicePtr,size,cudaMemcpyDeviceToHost));
  mat->isHostSide = 1;
}

float getCudaMatrixElementImpl(Matrix* self, int i, int  j)
{
  if (!self->isHostSide)
  {
    printf("\n### GPU WARNING ###\n");
    printf("Matrix was on device when trying to get.");
    printf("\n###################\n");
    copyDeviceToHostCudaMatrix(self);
  }
  return self->hostPtr[IDX2C(i,j,self->shape[1])];
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
    printf("\n### GPU WARNING ###\n");
    printf("Matrix was on device when trying to set!\n");
    printf("\n###################\n");
    copyDeviceToHostCudaMatrix(self);
  }
  self->hostPtr[IDX2C(i,j,self->shape[1])] = x;
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

Matrix* newEmptyCudaMatrixImpl(int rows, int columns)
{
  Matrix* m = (Matrix*)malloc(sizeof(Matrix));
  float* h_data = (float*)malloc(sizeof(float)*columns*rows);
  memset(h_data,0,sizeof(float)*columns*rows);
  float* d_data;
  cudaErrCheck(cudaMalloc(&d_data,sizeof(float)*columns*rows));
  int* shape = (int*)malloc(sizeof(int)*2);
  shape[0] = rows;
  shape[1] = columns;

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

  return m;
}

Matrix* newCudaMatrixImpl(float* data, int rows, int columns)
{
  Matrix* m = newEmptyCudaMatrixImpl(rows,columns);
  free(m->hostPtr);
  m->hostPtr = data;
  return m;
}
//############ BEGIN MATH FUNCS ##################
//ADDITION
void addCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
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
  copyDeviceToDeviceCudaMatrix(B,C);
  //cublasSetStream(_cublasHandle,self->stream);
  float a = 1;
  cublasErrCheck(cublasSaxpy(_cublasHandle,A->shape[0]*A->shape[1],&a,A->devicePtr,1,C->devicePtr,1));
}

//SUBTRACTION
void subtractCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
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
  copyDeviceToDeviceCudaMatrix(A,C);
  //cublasSetStream(_cublasHandle,self->stream);
  float a = -1;
  cublasErrCheck(cublasSaxpy(_cublasHandle,A->shape[0]*A->shape[1],&a,B->devicePtr,1,C->devicePtr,1));
}

//MIN
int* minRowsCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }

  int *idx = (int*)malloc(sizeof(int)*A->shape[0]);
  for (int i = 0; i < A->shape[0]; i++)
  {
    cublasErrCheck(cublasIsamin(_cublasHandle, A->shape[1],
                            A->devicePtr+(i*A->shape[1]), 1, idx+i));
    idx[i] -= 1;
  }
  return idx;
}

float maxValCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  int c = self->maxIdx(self,A);
  int* idx = C2IDX(c,A->shape[1]);
  return A->getElement(A,idx[0],idx[1]);
}

int maxIdxCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }

  int idx = 0;
  cublasErrCheck(cublasIsamax(_cublasHandle, A->shape[0]*A->shape[1],
                            A->devicePtr, 1, &idx));
  idx -= 1;
  return idx;
}

void powCudaMatrixImpl(MatrixUtil* self, Matrix* A, float k, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(1024,A->shape[1]*A->shape[0]);
  dim3 bdim(bdimX);
  dim3 gdim(A->shape[1]*A->shape[0]/bdimX + 1);
  PowMatrixKernel<<<gdim,bdim>>>(A->devicePtr,k,C->devicePtr,A->shape[0],A->shape[1]);
}

void transposeCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  int bdimX = fmin(32,A->shape[1]);
  int bdimY = fmin(32,A->shape[0]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape[0]/bdimX+1,A->shape[1]/bdimY + 1);
  TransposeMatrixKernel<<<gdim,bdim>>>(A->devicePtr,C->devicePtr,A->shape[0],A->shape[1]);
}

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

  Matrix* C = self->newEmptyMatrix(A->shape[0],B->shape[1]);

  self->subtract(self,A,B,C);

  float retval;
  cublasErrCheck(cublasSnrm2(_cublasHandle,A->shape[0]*A->shape[1],C->devicePtr,1,&retval));
  return retval;
}

//MULTIPLYCONST
void multiplyConstCudaMatrixImpl(MatrixUtil* self, Matrix* A, float b, Matrix* C)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (C->isHostSide)
  {
    copyHostToDeviceCudaMatrix(C);
  }
  //cublasSetStream(_cublasHandle,self->stream);
  cublasErrCheck(cublasSaxpy(_cublasHandle,A->shape[0]*A->shape[1],&b,A->devicePtr,1,C->devicePtr,1));
}

//DOT
void dotCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
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
  //cublasSetStream(_cublasHandle,self->stream);
  float alpha = 1;
  float beta = 0;
  int lda, tda, tdb;
  cublasOperation_t opA, opB;
  if (A->T)
  {
    opA = CUBLAS_OP_T;
    lda = A->shape[1];
    tda = A->shape[0];
  } else
  {
    opA = CUBLAS_OP_N;
    lda = A->shape[0];
    tda = A->shape[1];
  }
  if (B->T)
  {
    opB = CUBLAS_OP_T;
    tdb = B->shape[0];
  } else
  {
    opB = CUBLAS_OP_N;
    tdb = B->shape[1];
  }
  cublasErrCheck(cublasSgemm(_cublasHandle,
                           opA, opB,
                           lda, tdb, tda,
                           &alpha,
                           A->devicePtr, A->shape[0],
                           B->devicePtr, B->shape[0],
                           &beta,
                           C->devicePtr, C->shape[0]));
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
  int bdimX = fmin(32,B->shape[0]);
  int bdimY = fmin(32,A->shape[0]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(B->shape[0]/bdimX + 1,A->shape[0]/bdimY + 1);
  FeatureDistanceMatrixKernel<<<gdim,bdim>>>(A->devicePtr,A->shape[0],B->devicePtr,B->shape[0],C->devicePtr,A->shape[1]);
}

void makeCrossMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* Ax)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (Ax->isHostSide)
  {
    copyHostToDeviceCudaMatrix(Ax);
  }
  Cross3X3MatrixKernel<<<1,1>>>(A->devicePtr,Ax->devicePtr);
}

//CROSS
void cross3X3MatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Matrix* C)
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
  Matrix* Bx = self->newEmptyMatrix(3,3);
  self->makeCrossMatrix(self,B,Bx);
  float alpha = 1;
  float beta = 0;
  cublasErrCheck(cublasSgemv(_cublasHandle, CUBLAS_OP_N, 3, 3, &alpha, Bx->devicePtr, 3, A->devicePtr, 1, &beta, C->devicePtr, 1));
  Bx->free(Bx);
}

//inv
void invCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* Ainv)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (Ainv->isHostSide)
  {
    copyHostToDeviceCudaMatrix(Ainv);
  }

  float* Acopy;
  size_t size = sizeof(float)*A->shape[0]*A->shape[1];
  cudaErrCheck(cudaMalloc(&Acopy,size));
  cudaErrCheck(cudaMemcpy(Acopy,A->devicePtr,size,cudaMemcpyDeviceToDevice));

  int n = A->shape[0];
  int Lwork;

  //GET BUFFER size
  cusolverErrCheck(cusolverDnSgetrf_bufferSize(_cusolverHandle,
                      n,
                      n,
                      Acopy,
                      n,
                      &Lwork));

  //Create Workspace
  float* workspace;
  cudaErrCheck(cudaMalloc(&workspace,Lwork));

  //Prepare LU decomposition
  int* devIpiv;
  cudaErrCheck(cudaMalloc(&devIpiv,sizeof(int)*n));

  int* devInfo;
  cudaErrCheck(cudaMalloc(&devInfo,sizeof(int)));

  //DECOMPOSE
  cusolverErrCheck(cusolverDnSgetrf(_cusolverHandle,
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
    cudaErrCheck(cudaMemcpy(h_info,devInfo,sizeof(int),cudaMemcpyDeviceToHost));
    printf("LU DECOMPOSITION INFO: %i\n",h_info[0]);
  }

  //right hand sides
  float B[] = {1,0,0,0,1,0,0,0,1};
  float *d_B;
  cudaErrCheck(cudaMalloc(&d_B,size));
  cudaErrCheck(cudaMemcpy(d_B,&B,size,cudaMemcpyHostToDevice));

  //solve
  cusolverErrCheck(cusolverDnSgetrs(_cusolverHandle,
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
    cudaErrCheck(cudaMemcpy(h_info,devInfo,sizeof(int),cudaMemcpyDeviceToHost));
    printf("SOLVE INFO: %i\n",h_info[0]);
  }

  cudaErrCheck(cudaMemcpy(Ainv->devicePtr,d_B,size,cudaMemcpyDeviceToDevice));
  //printf("Copied");
  cudaErrCheck(cudaFree(workspace));
  cudaErrCheck(cudaFree(devIpiv));
  cudaErrCheck(cudaFree(devInfo));
  cudaErrCheck(cudaFree(Acopy));
  cudaErrCheck(cudaFree(d_B));
  //printf("RETURNING!\n");
}
//region = i,j,rows,cols
void copyCudaMatrixImpl(MatrixUtil* self, Matrix* A, Matrix* B, Rect size, Point2 Aidx, Point2 Bidx)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }
  if (B->isHostSide)
  {
    copyHostToDeviceCudaMatrix(B);
  }
  int bdimX = fmin(32,size.width);
  int bdimY = fmin(32,size.height);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(size.width/bdimX + 1,size.height/bdimY + 1);
  CopyMatrixKernel<<<gdim,bdim>>>(A->devicePtr,A->shape[0],A->shape[1],Aidx,B->devicePtr,B->shape[0],B->shape[1],Bidx,size);
}

void SetCUDAMatrixUtilStream(cudaStream_t stream)
{
    cublasSetStream(_cublasHandle, stream);
    cusolverDnSetStream(_cusolverHandle, stream);
}

void InitCUDAHandles(int device)
{
  cudaSetDevice(device);

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  _cublasHandle = cublasHandle;

  cusolverDnHandle_t cusolverHandle;
  cusolverDnCreate(&cusolverHandle);
  _cusolverHandle = cusolverHandle;

  //cudaStream_t stream;
  //cudaStreamCreate(&stream);
  //SetCUDAMatrixUtilStream(stream);
}

void pprintCudaMatrixImpl(MatrixUtil* self, Matrix* A, char* label)
{
  printf("\n\n################################################");
  printf("\n%s:\n\n",label);
  if (!A->isHostSide)
  {
    copyDeviceToHostCudaMatrix(A);
  }
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


  DLLEXPORT MatrixUtil* GetMatrixUtil()
{
  MatrixUtil* cudaMatrixUtil = (MatrixUtil*)malloc(sizeof(MatrixUtil));
  InitCUDAHandles(1);

  cudaMatrixUtil->newEmptyMatrix = newEmptyCudaMatrixImpl;
  cudaMatrixUtil->newMatrix = newCudaMatrixImpl;
  cudaMatrixUtil->copy = copyCudaMatrixImpl;
  cudaMatrixUtil->pprint = pprintCudaMatrixImpl;

  cudaMatrixUtil->add = addCudaMatrixImpl;
  cudaMatrixUtil->subtract = subtractCudaMatrixImpl;
  cudaMatrixUtil->dot = dotCudaMatrixImpl;
  cudaMatrixUtil->multiplyConst = multiplyConstCudaMatrixImpl;
  cudaMatrixUtil->distance = distanceCudaMatrixImpl;
  cudaMatrixUtil->makeCrossMatrix = makeCrossMatrixImpl;
  cudaMatrixUtil->cross = cross3X3MatrixImpl;
  cudaMatrixUtil->inv = invCudaMatrixImpl;
  cudaMatrixUtil->maxIdx = maxIdxCudaMatrixImpl;
  cudaMatrixUtil->maxVal = maxValCudaMatrixImpl;
  cudaMatrixUtil->minRows = minRowsCudaMatrixImpl;
  cudaMatrixUtil->pow = powCudaMatrixImpl;
  cudaMatrixUtil->featureDistance = featureDistanceCudaMatrixImpl;
  cudaMatrixUtil->transpose = transposeCudaMatrixImpl;
/*
  cudaMatrixUtil->multiply = multiplyCudaMatrixImpl;
  cudaMatrixUtil->divide = divideCudaMatrixImpl;
  cudaMatrixUtil->divideConst = divideConstCudaMatrixImpl;
  cudaMatrixUtil->pow = powCudaMatrixImpl;
  cudaMatrixUtil->convolve = convolveCudaMatrixImpl;
  cudaMatrixUtil->sqrt = sqrtCudaMatrixImpl;
  cudaMatrixUtil->exp = expCudaMatrixImpl;
  cudaMatrixUtil->log = logCudaMatrixImpl;
  cudaMatrixUtil->arctan = arctanCudaMatrixImpl;



  cudaMatrixUtil->solve = solveCudaMatrixImpl;
  cudaMatrixUtil->lstsq = lstsqCudaMatrixImpl;

  //cudaMatrixUtil->ceil = ceilCudaMatrixImpl;
  //cudaMatrixUtil->floor = floorCudaMatrixImpl;
  //cudaMatrixUtil->abs = absCudaMatrixImpl;
  cudaMatrixUtil->isEqual = isEqualCudaMatrixImpl;
  cudaMatrixUtil->convolve = convolveCudaMatrixImpl;*/

  return cudaMatrixUtil;
}
#ifdef __cplusplus
  }
#endif
