#include <core.h>

#ifdef __cplusplus
extern "C" {
#endif

int IDX2C(Point2 index, Shape shape)
{
  return (index.y*shape.width)+index.x;
}

Point2 C2IDX(int i, Shape shape)
{
  int y = i/shape.width;
  int x = i-(y*shape.width);
  Point2 retval = {x,y};
  return retval;
}

int SHAPE2LEN(Shape shape)
{
  return shape.width*shape.height;
}

void make_launch_parameters(Shape shape, int dim, dim3 *bdim, dim3 *gdim)
{
  if (dim == 1)
  {
    int len = SHAPE2LEN(shape);
    int dim = fmin(THREADS_PER_BLOCK,len);
    dim3 b = {dim,1,1};
    dim3 g = {len/dim + 1,1,1};
	  *bdim = b;
	  *gdim = g;

  } else if (dim == 2)
  {
    int x_len = shape.width;
    int y_len = shape.height;
    int x_dim = fmin(sqrt(THREADS_PER_BLOCK),x_len);
    int y_dim = fmin(sqrt(THREADS_PER_BLOCK),y_len);
    dim3 b = {x_dim,y_dim,1};
    dim3 g = {x_len/x_dim + 1, y_len/y_dim + 1,1};
	  *bdim = b;
	  *gdim = g;
  }
}

#ifdef __cplusplus
}
#endif
