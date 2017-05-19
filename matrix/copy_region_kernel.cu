#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void copy_region_kernel(float *a, float *b, Point2 a_idx, Point2 b_idx, Shape region, Shape a_shape, Shape b_shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  if (y<region.height && x < region.width)
  {
    Point2 dest_idx = {x+b_idx.x, y+b_idx.y};
    Point2 from_idx = {x+a_idx.x,y+a_idx.y};
    b[IDX2C_kernel(dest_idx,b_shape)] = a[IDX2C_kernel(from_idx,a_shape)];
  }
}

#ifdef __cplusplus
}
#endif
