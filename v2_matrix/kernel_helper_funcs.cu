#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__device__ int IDX2C_kernel(Point2 index, Shape shape)
{
  return (index.y*shape.width)+index.x;
}

__device__ Point2 C2IDX_kernel(int i, Shape shape)
{
  int y = i/shape.width;
  int x = i-(y*shape.width);
  Point2 retval = {x,y};
  return retval;
}

__device__ int SHAPE2LEN_kernel(Shape shape)
{
  return shape.width*shape.height;
}

#ifdef __cplusplus
}
#endif
