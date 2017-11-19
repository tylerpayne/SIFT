#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void fill_kernel(float *a, float b, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    a[x] = b;
  }
}

#ifdef __cplusplus
}
#endif
