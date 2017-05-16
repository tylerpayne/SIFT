#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

int free_matrix(Matrix *m)
{
  free(m->host_ptr);
  if (m->dev_ptr != NULL)
      cuda_safe_call(cudaFree(m->dev_ptr));
  return 0;
}
#ifdef __cplusplus
}
#endif
