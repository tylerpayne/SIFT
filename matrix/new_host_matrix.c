#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

void new_host_matrix(Matrix **new_matrix, float* host_ptr, Shape shape)
{
  *new_matrix = (Matrix*)malloc(sizeof(Matrix));
  (*new_matrix)->host_ptr = host_ptr;
  (*new_matrix)->isHostSide = TRUE;
  (*new_matrix)->shape = shape;
  (*new_matrix)->T = FALSE;
}

#ifdef __cplusplus
}
#endif
