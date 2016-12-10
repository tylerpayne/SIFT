#include <cuda_runtime.h>

__device__ int IDX2CKernel(int i, int j, int td)
{
  return (i*td)+j;
}

//ADDITION
__global__ void MatAdd(float* A, float* B, float* C,int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = A[IDX2CKernel(row,col,td)] + B[IDX2CKernel(row,col,td)];
  }
}

__global__ void MatAddConst(float* A, float B, float* C,int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = A[IDX2CKernel(row,col,td)] + B;
  }
}
//SUBTRACTION
__global__ void MatSubtract(float* A, float* B, float* C,int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = A[IDX2CKernel(row,col,td)] - B[IDX2CKernel(row,col,td)];
  }
}

__global__ void MatSubtractConst(float* A, float B, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = A[IDX2CKernel(row,col,td)] - B;
  }
}

//MULTIPLICATION
__global__ void MatMult(float* A, float* B, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = A[IDX2CKernel(row,col,td)] * B[IDX2CKernel(row,col,td)];
  }
}

__global__ void MatMultConst(float* A, float B, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = A[IDX2CKernel(row,col,td)]*B;
  }
}

//Division
__global__ void MatDivide(float* A, float* B, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
   C[IDX2CKernel(row,col,td)] = A[IDX2CKernel(row,col,td)] / B[IDX2CKernel(row,col,td)];
  }
}

__global__ void MatDivideConst(float* A, float B, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = A[IDX2CKernel(row,col,td)] / B;
  }
}

//pow
__global__ void MatPow(float* A, float B, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = powf(A[IDX2CKernel(row,col,td)],B);
  }
}

//sqrt
__global__ void MatSqrt(float* A, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = sqrtf(A[IDX2CKernel(row,col,td)]);
  }
}

//exp
__global__ void MatExp(float* A, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = expf(A[IDX2CKernel(row,col,td)]);
  }
}

//abs
__global__ void MatAbs(float* A, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = abs(A[IDX2CKernel(row,col,td)]);
  }
}

//arcctan
__global__ void MatArctan(float* A, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = atanf(A[IDX2CKernel(row,col,td)]);
  }
}

//log
__global__ void MatLog(float* A, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    C[IDX2CKernel(row,col,td)] = log(A[IDX2CKernel(row,col,td)]);
  }
}

//isEqual
__global__ void MatisEqual(float* A, float* B, float* C, int ld, int td)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (row < ld && col < td)
  {
    if (B[IDX2CKernel(row,col,td)] == A[IDX2CKernel(row,col,td)])
    {
      C[IDX2CKernel(row,col,td)] = 1;
    } else
    {
      C[IDX2CKernel(row,col,td)] = 0;
    }
  }
}

extern __shared__ float sharedFloatArray[];
//CONVOLVE
__global__ void MatConvolve(float* A, float* B, float* C, int ald, int atd, int bld, int btd)
{

  float* Adata = (float*)&sharedFloatArray[btd*bld];
  float* Bdata = (float*)&sharedFloatArray[btd*bld];

  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (row < ald && col < atd)
  {
    int radius = btd/2;
    for (int i = 0; i < btd; i++)
    {
      for (int j = 0; j < btd; j++)
      {
        int ii = i-radius;
        int jj = j-radius;
        Bdata[IDX2CKernel(i,j,btd)] = B[IDX2CKernel(i,j,btd)];
        Adata[IDX2CKernel(row+ii,col+jj,atd)] = A[IDX2CKernel(row+ii,col+jj,atd)];
      }
    }
    __syncthreads();
    C[IDX2CKernel(row,col,atd)] = 0.0;
    for (int i = 0; i < btd; i++)
    {
      for (int j = 0; j < btd; j++)
      {
        int ii = i-radius;
        int jj = j-radius;
        if (row+ii >= 0 && row+ii < ald && col+jj >= 0 && col+jj < atd)
        {
          C[IDX2CKernel(row,col,atd)] += Adata[IDX2CKernel(row+ii,col+jj,atd)] * Bdata[IDX2CKernel(i,j,btd)];
        }
      }
    }
  }
}

// TRANSPOSE
__global__ void MatTranspose(float* A, float* C, int ald, int atd)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (row < ald && col < atd)
  {
    C[IDX2CKernel(col,row,ald)] = A[IDX2CKernel(row,col,atd)];
  }
}

//B IS TRANSPOSED
//i.e. atd = btd = cd
__global__ void MatDot(float* A, float* B, float* C, int ald, int cd, int bld)
{
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
//  C[IDX2CKernel(row,col,cd)] = 100;
  if (row < ald && col < bld)
  {
    float retval = 0;
    for (int i = 0; i < cd; i++)
    {
      retval += A[IDX2CKernel(row,i,cd)] * B[IDX2CKernel(col,i,cd)];
    }
    C[IDX2CKernel(row,col,cd)] = retval;
  }
}
