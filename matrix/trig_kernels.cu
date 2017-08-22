#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void cos_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = cosf(a[x]);
  }
}

__global__ void sin_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = sinf(a[x]);
  }
}

__global__ void tan_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = tanf(a[x]);
  }
}

__global__ void acos_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = acosf(a[x]);
  }
}

__global__ void asin_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = asinf(a[x]);
  }
}

__global__ void atan_kernel(float *a, float *b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    b[x] = atanf(a[x]);
  }
}

__global__ void atan2_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = atan2f(a[x],b[x]);
  }
}

__global__ void hypot_kernel(float *a, float *b, float *c, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    c[x] = hypotf(a[x],b[x]);
  }
}

#ifdef __cplusplus
}
#endif
