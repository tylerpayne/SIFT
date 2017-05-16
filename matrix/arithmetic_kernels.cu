#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void add_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] + b[x];
  }
}

__global__ void divide_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] / b[x];
  }
}

__global__ void multiply_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] * b[x];
  }
}

__global__ void subtract_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] - b[x];
  }
}

#ifdef __cplusplus
}
#endif
