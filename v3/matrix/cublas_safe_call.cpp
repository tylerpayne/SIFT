#include <core.h>

void cublas_safe_call(cublasStatus_t err)
{
  if (err != CUBLAS_STATUS_SUCCESS)
    printf("CUBLAS ERROR: %i\n",err);
    //exit(1);
}
