#include <chai.h>

namespace chai {

  int IDX2C(Tuple<int> index, Tuple<int> shape)
  {
    return (index.components[1]*shape.components[0])+index.components[0];
  }

  Tuple<int> C2IDX(int i, Tuple<int> shape)
  {
    int y = i/shape.components[0];
    int x = i-(y*shape.components[0]);
    Tuple<int> retval({x,y});
    return retval;
  }

  void make_launch_parameters(Tuple<int> shape, int dim, dim3 *bdim, dim3 *gdim)
  {
    if (dim == 1)
    {
      int len = shape.prod();
      int dim = min(THREADS_PER_BLOCK,len);
      dim3 b = {dim,1,1};
      dim3 g = {len/dim + 1,1,1};
  	  *bdim = b;
  	  *gdim = g;
    } else if (dim == 2)
    {
      int x_len = shape.components[0];
      int y_len = shape.components[1];
      int x_dim = fmin(sqrt(THREADS_PER_BLOCK),x_len);
      int y_dim = fmin(sqrt(THREADS_PER_BLOCK),y_len);
      dim3 b = {x_dim,y_dim,1};
      dim3 g = {x_len/x_dim + 1, y_len/y_dim + 1,1};
  	  *bdim = b;
  	  *gdim = g;
    }
  }

}
