#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void gt_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] > b[x];
  }
}

__global__ void gte_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] >= b[x];
  }
}

__global__ void lt_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] < b[x];
  }
}

__global__ void lte_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] <= b[x];
  }
}

__global__ void eq_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = a[x] == b[x];
  }
}

#ifdef __cplusplus
}
#endif
