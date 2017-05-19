#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

int dog(Matrix *a, Matrix *out, Shape kernel_shape, float s1, float s2)
{
  memassert(a,DEVICE);
  Matrix *g1, *g2, *diff;
  new_empty_matrix(&diff,kernel_shape);
  generate_gaussian(&g1,kernel_shape,s1,s1);
  generate_gaussian(&g2,kernel_shape,s2,s2);

  subtract(g1,g2,diff);
  convolve(a,diff,out);

  free_matrix(g1);
  free_matrix(g2);
  free_matrix(diff);

  return 0;
}

#ifdef __cplusplus
}
#endif
