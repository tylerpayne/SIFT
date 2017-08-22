#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

void new_empty_matrix(Matrix **new_matrix, Shape shape)
{
  *new_matrix = (Matrix*)malloc(sizeof(Matrix));
  size_t size = sizeof(float)*shape.width*shape.height;
  float *host_ptr = (float*)malloc(size);
  memset(host_ptr,0,size);
  (*new_matrix)->host_ptr = host_ptr;
  (*new_matrix)->isHostSide = TRUE;
  (*new_matrix)->shape = shape;
  (*new_matrix)->T = FALSE;
}

#ifdef __cplusplus
}
#endif
