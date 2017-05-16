#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

void new_device_matrix(Matrix **new_matrix, float* dev_ptr, Shape shape)
{
  *new_matrix = (Matrix*)malloc(sizeof(Matrix));
  (*new_matrix)->host_ptr = (float*)malloc(sizeof(float)*SHAPE2LEN(shape));
  (*new_matrix)->dev_ptr = dev_ptr;
  (*new_matrix)->isHostSide = FALSE;
  (*new_matrix)->shape = shape;
  (*new_matrix)->T = FALSE;
}

#ifdef __cplusplus
}
#endif
