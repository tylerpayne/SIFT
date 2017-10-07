#include <matrix.h>


__device__ int IDX2C_kernel(Tuple<int> index, Tuple<int> shape)
{
  return (index.components[1]*shape.components[0])+index.components[0];
}

__device__ Point2 C2IDX_kernel(int i, Tuple<int> shape)
{
  int width = shape.components[0];
  int height = shape.components[1];
  int y = i/width;
  int x = i-(y*width);
  Point2 retval = {x,y};
  return retval;
}

__device__ int SHAPE2LEN_kernel(Tuple shape)
{
  return shape.width*shape.height;
}
