#include <matrix.h>

float get_element(Matrix *a, Point2 id)
{
  memassert(a, HOST);
  return a->host_ptr[IDX2C(id,a->shape)];
}
