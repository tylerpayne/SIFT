#include <math_constants.h>

__device__ int IDX2CKernel(int i, int j, int td)
{
  return (i*td)+j;
}

__device__ Point2 C2IDXKernel(int c, int td)
{
  int row = c/td;
  int col = c - (row*td)
  Point2 ret = {row,col};
  return ret;
}

__global__ void MultiplyElemWiseKernel(float* A, float* B, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = A[i]*B[i];
  }
}

__global__ void DivideElemWiseKernel(float* A, float* B, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = A[i]/B[i];
  }
}



__global__ void PowMatrixKernel(float* A, float b, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = powf(A[i],b);
  }
}

__global__ void AbsMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = fabsf(A[i]);
  }
}

__global__ void SqrtMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = sqrtf(A[i]);
  }
}

__global__ void CosMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = cosf(A[i]);
  }
}

__global__ void SinMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = sinf(A[i]);
  }
}

__global__ void TanMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = tanf(A[i]);
  }
}

__global__ void ArccosMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = acosf(A[i]);
  }
}

__global__ void ArcsinMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = asinf(A[i]);
  }
}

__global__ void ArctanMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = atanf(A[i]);
  }
}

__global__ void LogMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = logf(A[i]);
  }
}

__global__ void ExpMatrixKernel(float* A, float* C, Shape shape)
{
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < shape.width*shape.height)
  {
    C[i] = expf(A[i]);
  }
}


__global__ void TransposeMatrixKernel(float* A, float* C, int ld, int td)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (y<ld && x < td)
  {
    C[IDX2CKernel(x,y,ld)] = A[IDX2CKernel(y,x,td)];
  }
}

__global__ void FeatureDistanceMatrixKernel(float* A, int lda, float* B, int ldb, float* C, int nDim)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (y < lda && x < ldb)
  {
    float accum = 0;
    for (int i = 0; i < nDim; i++)
    {
      float diff = B[IDX2CKernel(x,i,nDim)] - A[IDX2CKernel(y,i,nDim)];
      accum += diff*diff;
    }
    C[IDX2CKernel(y,x,ldb)] = sqrtf(accum);
  }
}

__global__ void Cross3X3MatrixKernel(float* A, float* Ax)
{
  Ax[IDX2CKernel(0,1,3)] = -1.0*A[2];
  Ax[IDX2CKernel(0,2,3)] = A[1];
  Ax[IDX2CKernel(1,2,3)] = -1.0*A[0];

  Ax[IDX2CKernel(1,0,3)] = A[2];
  Ax[IDX2CKernel(2,0,3)] = -1.0*A[1];
  Ax[IDX2CKernel(2,1,3)] = A[0];
}

__global__ void CopyMatrixKernel(float* A, int lda, int tda, Point2 Aidx, float* B, int ldb, int tdb, Point2 Bidx, Rect size)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int thisA, thisB;
  thisA = (Aidx.y + y)*tda + (Aidx.x + x);
  thisB = (Bidx.y + y)*tdb + (Bidx.x + x);
  if (y < size.shape.height && x < size.shape.width)
  {
    B[thisB] = A[thisA];
  }

}

/*
//#####################
//#### OUTDATED! ######
//#####################


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

*/
