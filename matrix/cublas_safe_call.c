#include <core.h>

#ifdef __cplusplus
extern "C" {
#endif

void cublas_safe_call(cublasStatus_t err)
{
  if (err != CUBLAS_STATUS_SUCCESS)
    printf("CUBLAS ERROR: %i\n",err);
    //exit(1);
}

#ifdef __cplusplus
}
#endif
