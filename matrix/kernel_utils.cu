#include <chai.h>

namespace chai {
  namespace cuda
  {
    __device__ int IDX2C_kernel(int index[2], int shape[2])
    {
      return (index[1]*shape[0])+index[0];
    }

    __device__ void C2IDX_kernel(int i, int shape[2], int *r, int *c)
    {
      int width = shape[0];
      *r = i/width;
      *c = i-(*r*width);
    }

    __device__ int tuple_product(int_tuple shape)
    {
      int retval = shape.components[0];
      for (int i = 1; i < shape.length; i++)
      {
        retval *= shape.components[i];
      }
      return retval;
    }
  }
}
