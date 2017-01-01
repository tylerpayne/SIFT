#include "MatrixUtil.h"
#include "MathKernels.cu"

cudaError_t cudaErr;
cublasStatus_t cublasStat;

void cublasErrCheck(cublasStatus_t stat)
{
  if (stat != CUBLAS_STATUS_SUCCESS)
  {
    printf("\nCUBLAS ERR: %i\n",stat);
  }
}

void freeCudaMatrixDeviceMemory(Matrix* mat)
{
  cudaFree(mat->devicePtr);
}

void copyDeviceToDeviceCudaMatrix(MatrixUtil* self, Matrix* A, Matrix* B)
{
    cudaSetDevice(self->deviceId);
    size_t size = sizeof(float)*A->shape[0]*A->shape[1];
    cudaMemcpyAsync(B->devicePtr,A->devicePtr,size,cudaMemcpyDeviceToDevice,self->stream);
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
    cudaMemcpyAsync(mat->devicePtr,mat->hostPtr,size,cudaMemcpyHostToDevice,self->stream);
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
  cudaMemcpyAsync(mat->hostPtr,mat->devicePtr,size,cudaMemcpyDeviceToHost,self->stream);
  mat->isHostSide = 1;
}

float getCudaMatrixElementImpl(Matrix* self, int i, int  j)
{
  if (!self->isHostSide)
  {
    printf("\n### GPU WARNING ###\n");
    printf("Matrix was on device when trying to get.");
    printf("\n###################\n");
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

Matrix* newEmptyCudaMatrixImpl(int width, int height)
{
  Matrix* m = (Matrix*)malloc(sizeof(Matrix));
  float* h_data = (float*)malloc(sizeof(float)*width*height);
  memset(h_data,0,sizeof(float)*width*height);
  float* d_data;
  cudaMalloc(&d_data,sizeof(float)*width*height);
  int* shape = (int*)malloc(sizeof(int)*2);
  shape[0] = height;
  shape[1] = width;

  m->shape = shape;
  m->hostPtr = h_data;
  m->devicePtr = d_data;
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
  copyDeviceToDeviceCudaMatrix(self,B,C);
  //cublasSetStream(self->cublasHandle,self->stream);
  float a = 1;
  cublasErrCheck(cublasSaxpy(self->cublasHandle,A->shape[0]*A->shape[1],&a,A->devicePtr,1,C->devicePtr,1));
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
  copyDeviceToDeviceCudaMatrix(self,A,C);
  //cublasSetStream(self->cublasHandle,self->stream);
  float a = -1;
  cublasErrCheck(cublasSaxpy(self->cublasHandle,A->shape[0]*A->shape[1],&a,B->devicePtr,1,C->devicePtr,1));
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
  //cublasSetStream(self->cublasHandle,self->stream);
  cublasErrCheck(cublasSaxpy(self->cublasHandle,A->shape[0]*A->shape[1],&b,A->devicePtr,1,C->devicePtr,1));
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
  //cublasSetStream(self->cublasHandle,self->stream);
  float alpha = 1;
  float beta = 0;
  cublasErrCheck(cublasSgemm(self->cublasHandle,A->T,B->T,A->shape[0],B->shape[1],A->shape[1],&alpha,A->devicePtr,A->shape[0],B->devicePtr,B->shape[0],&beta,C->devicePtr,C->shape[0]));
}

void SetCUDAMatrixUtilStream(MatrixUtil* self, cudaStream_t stream)
{
    self->stream = stream;
    cublasSetStream(self->cublasHandle, stream);
}

void SetCUDAMatrixUtilDevice(MatrixUtil* self, int device)
{
  self->deviceId = device;
  cudaSetDevice(device);

  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  self->cublasHandle = cublasHandle;

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  SetCUDAMatrixUtilStream(self,stream);
}

void pprintCudaMatrixImpl(MatrixUtil* self, Matrix* A, char* label)
{
  printf("\n\n################################################");
  printf("\n%s:\n\n",label);
  copyDeviceToHostCudaMatrix(self,A);
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

MatrixUtil* GetCUDAMatrixUtil(int device)
{
  MatrixUtil* cudaMatrixUtil = (MatrixUtil*)malloc(sizeof(MatrixUtil));
  SetCUDAMatrixUtilDevice(cudaMatrixUtil,device);

  cudaMatrixUtil->newEmptyMatrix = newEmptyCudaMatrixImpl;
  cudaMatrixUtil->pprint = pprintCudaMatrixImpl;
  cudaMatrixUtil->newMatrix = newCudaMatrixImpl;
  cudaMatrixUtil->add = addCudaMatrixImpl;
  cudaMatrixUtil->subtract = subtractCudaMatrixImpl;
  cudaMatrixUtil->dot = dotCudaMatrixImpl;
  cudaMatrixUtil->multiplyConst = multiplyConstCudaMatrixImpl;
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
  cudaMatrixUtil->transpose = transposeCudaMatrixImpl;

  cudaMatrixUtil->inv = invCudaMatrixImpl;
  cudaMatrixUtil->solve = solveCudaMatrixImpl;
  cudaMatrixUtil->lstsq = lstsqCudaMatrixImpl;

  //cudaMatrixUtil->ceil = ceilCudaMatrixImpl;
  //cudaMatrixUtil->floor = floorCudaMatrixImpl;
  //cudaMatrixUtil->abs = absCudaMatrixImpl;
  cudaMatrixUtil->isEqual = isEqualCudaMatrixImpl;
  cudaMatrixUtil->convolve = convolveCudaMatrixImpl;*/
//TODO: cross

  return cudaMatrixUtil;
}
