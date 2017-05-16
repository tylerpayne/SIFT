#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void sqrt_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = sqrtf(a[x]);
  }
}

__global__ void abs_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = fabsf(a[x]);
  }
}

__global__ void exp_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = expf(a[x]);
  }
}

__global__ void log_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = logf(a[x]);
  }
}

__global__ void pow_kernel(float *a, float b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = powf(a[x],b);
  }
}

#ifdef __cplusplus
}
#endif
