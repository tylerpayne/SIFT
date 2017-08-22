#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void gt_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    float diff = a[x]-b[x];
    c[x] = (float)(diff > 0.01);
  }
}

__global__ void gte_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    float diff = a[x]-b[x];
    c[x] = (float)(diff > -0.01);
  }
}

__global__ void lt_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    float diff = a[x]-b[x];
    c[x] = (float)(diff < -0.01);
  }
}

__global__ void lte_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    float diff = a[x]-b[x];
    c[x] = (float)(diff < 0.01);
  }
}

__global__ void eq_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    float diff = a[x]-b[x];
    c[x] = (float)(diff > -0.01 && diff < 0.01);
  }
}

#ifdef __cplusplus
}
#endif
