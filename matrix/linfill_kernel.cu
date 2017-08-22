#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void linfill_kernel(float *a, float from, float step, Shape shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    float b = from + ((float)x)*step;
    a[x] = b;
  }
}

#ifdef __cplusplus
}
#endif
