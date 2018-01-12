#include <chai.h>

namespace chai {

int idx2c(tuple<int> index, tuple<int> shape) {
  return (index.components[0] * shape.components[1]) + index.components[1];
}

int idx2c(int i, int j, tuple<int> shape) {
  return (i * shape.components[1]) + j;
}

tuple<int> c2idx(int i, tuple<int> shape) {
  int y = i / shape.components[1];
  int x = i - (y * shape.components[1]);
  tuple<int> retval({x, y});
  return retval;
}

void make_launch_parameters(tuple<int> shape, int dim, dim3 *bdim,
                            dim3 *gdim) {
  if (dim == 1) {
    unsigned int len = shape.prod();
    unsigned int dim = min(THREADS_PER_BLOCK, len);
    dim3 b = {dim, 1, 1};
    dim3 g = {len / dim + 1, 1, 1};
    *bdim = b;
    *gdim = g;
  } else if (dim == 2) {
    unsigned int x_len = shape.components[1];
    unsigned int y_len = shape.components[0];
    unsigned int x_dim = fmin(sqrt(THREADS_PER_BLOCK), x_len);
    unsigned int y_dim = fmin(sqrt(THREADS_PER_BLOCK), y_len);
    dim3 b = {x_dim, y_dim, 1};
    dim3 g = {x_len / x_dim + 1, y_len / y_dim + 1, 1};
    *bdim = b;
    *gdim = g;
  }
}
}
