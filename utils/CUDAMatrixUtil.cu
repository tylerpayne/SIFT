#include <utils//MatrixUtil.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
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
    printf("CUDA ERR\n%s\n",cudaGetErrorString(stat));
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
  printf("matfree\n");
  if (m->devicePtr != NULL)
  {
    freeCudaMatrixDeviceMemory(m);
  }
  free(m->hostPtr);
  free(m);
}

DLLEXPORT void copyDeviceToDeviceCudaMatrix(Matrix* A, Matrix* B)
{
    size_t size = sizeof(float)*A->shape.width*A->shape.height;
    cudaErrCheck(cudaMemcpy(B->devicePtr,A->devicePtr,size,cudaMemcpyDeviceToDevice));
}

DLLEXPORT void copyHostToDeviceCudaMatrix(Matrix* mat)
{
    size_t size = sizeof(float)*mat->shape.width*mat->shape.height;
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
  size_t size = sizeof(float)*mat->shape.width*mat->shape.height;
  if (VERBOSITY > 3)
  {
    printf("\n\n### GPU WARNING ###\n");
    printf("Copying Device to Host");
    printf("\n###################\n\n");
  }
  cudaErrCheck(cudaMemcpy(mat->hostPtr,mat->devicePtr,size,cudaMemcpyDeviceToHost));
  mat->isHostSide = 1;
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
    cudaErrCheck(cudaMalloc(&data,s));
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
  cudaErrCheck(cudaMalloc(&d_data,sizeof(float)*shape.height*shape.width));

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

Matrix* newCudaMatrixImpl(float* data, Shape shape)
{
  Matrix* m = newEmptyCudaMatrixImpl(shape);
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
  float a = 1;
  cublasErrCheck(cublasSaxpy(_cublasHandle,A->shape.height*A->shape.width,&a,A->devicePtr,1,C->devicePtr,1));
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
  float a = -1;
  cublasErrCheck(cublasSaxpy(_cublasHandle,A->shape.height*A->shape.width,&a,B->devicePtr,1,C->devicePtr,1));
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
  cublasErrCheck(cublasSaxpy(_cublasHandle,A->shape.height*A->shape.width,&b,A->devicePtr,1,C->devicePtr,1));
}

//MIN
int* minRowsCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }

  int *idx = (int*)malloc(sizeof(int)*A->shape.height);
  for (int i = 0; i < A->shape.height; i++)
  {
    cublasErrCheck(cublasIsamin(_cublasHandle, A->shape.width,
                            A->devicePtr+(i*A->shape.width), 1, idx+i));
    idx[i] -= 1;
  }
  return idx;
}

float maxValCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  int c = self->maxIdx(self,A);
  Point2 idx = C2IDX(c,A->shape);
  return A->getElement(A,idx);
}

int maxIdxCudaMatrixImpl(MatrixUtil* self, Matrix* A)
{
  if (A->isHostSide)
  {
    copyHostToDeviceCudaMatrix(A);
  }

  int idx = 0;
  cublasErrCheck(cublasIsamax(_cublasHandle, A->shape.height*A->shape.width,
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
  int bdimX = fmin(1024,A->shape.width*A->shape.height);
  dim3 bdim(bdimX);
  dim3 gdim(A->shape.width*A->shape.height/bdimX + 1);
  PowMatrixKernel<<<gdim,bdim,0,_stream>>>(A->devicePtr,k,C->devicePtr,A->shape.height,A->shape.width);
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
  int bdimX = fmin(32,A->shape.width);
  int bdimY = fmin(32,A->shape.height);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(A->shape.height/bdimX+1,A->shape.width/bdimY + 1);
  TransposeMatrixKernel<<<gdim,bdim,0,_stream>>>(A->devicePtr,C->devicePtr,A->shape.height,A->shape.width);
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

  Matrix* C = self->newEmptyMatrix(A->shape);

  self->subtract(self,A,B,C);

  float retval;
  cublasErrCheck(cublasSnrm2(_cublasHandle,A->shape.height*A->shape.width,C->devicePtr,1,&retval));
  return retval;
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
  cublasErrCheck(cublasSgemm(_cublasHandle,
                           opA, opB,
                           lda, tdb, tda,
                           &alpha,
                           A->devicePtr, A->shape.height,
                           B->devicePtr, B->shape.height,
                           &beta,
                           C->devicePtr, C->shape.height));
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
  Cross3X3MatrixKernel<<<1,1,0,_stream>>>(A->devicePtr,Ax->devicePtr);
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
  Shape shape = {3,3};
  Matrix* Bx = self->newEmptyMatrix(shape);
  self->makeCrossMatrix(self,B,Bx);
  float alpha = 1;
  float beta = 0;
  cublasErrCheck(cublasSgemv(_cublasHandle, CUBLAS_OP_N, 3, 3, &alpha, Bx->devicePtr, 3, A->devicePtr, 1, &beta, C->devicePtr, 1));
  Bx->free(Bx);
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
  size_t size = sizeof(float)*A->shape.height*A->shape.width;
  cudaErrCheck(cudaMalloc(&Acopy,size));
  cudaErrCheck(cudaMemcpy(Acopy,A->devicePtr,size,cudaMemcpyDeviceToDevice));

  int n = A->shape.height;
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
  cudaErrCheck(cudaFree(workspace));
  cudaErrCheck(cudaFree(devIpiv));
  cudaErrCheck(cudaFree(devInfo));
  cudaErrCheck(cudaFree(Acopy));
  cudaErrCheck(cudaFree(d_B));
}

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
  int bdimX = fmin(32,size.shape.width);
  int bdimY = fmin(32,size.shape.height);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(size.shape.width/bdimX + 1,size.shape.height/bdimY + 1);
  CopyMatrixKernel<<<gdim,bdim,0,_stream>>>(A->devicePtr,A->shape.height,A->shape.width,Aidx,B->devicePtr,B->shape.height,B->shape.width,Bidx,size);
}

void SetCUDAMatrixUtilStream(cudaStream_t stream)
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
  printf("\n\n################################################");
  printf("\n%s:\n\n",label);
  if (!A->isHostSide)
  {
    copyDeviceToHostCudaMatrix(A);
  }
  for (int i = 0; i < A->shape.height; i++)
  {
    for (int j = 0;j< A->shape.width;j++)
    {
      Point2 idx = {j,i};
      printf("[ %f ]",A->getElement(A,idx));
    }
    //printf("|  %f",y->getElement(y,i,0));
    printf("\n");
  }
  printf("\n################################################\n\n");
}


  DLLEXPORT MatrixUtil* GetMatrixUtil()
{
  MatrixUtil* cudaMatrixUtil = (MatrixUtil*)malloc(sizeof(MatrixUtil));
  InitCUDAHandles();

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
