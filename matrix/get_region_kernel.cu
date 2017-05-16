#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void get_region_kernel(float *a, float *b, Shape shape, Rect region)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if (y<shape.height && x < shape.width)
  {
    Point2 dest_idx = {x, y};
    Point2 from_idx = {x+region.origin.x,y+region.origin.y};
    b[IDX2C_kernel(dest_idx,region.shape)] = a[IDX2C_kernel(from_idx,shape)];
  }
}

#ifdef __cplusplus
}
#endif
