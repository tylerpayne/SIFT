#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t _cublasHandle;

int argmin(Matrix *a, int *index)
{
  memassert(a,DEVICE);
  cublas_safe_call(
    cublasIsamin(_cublasHandle,
      SHAPE2LEN(a->shape),a->dev_ptr, sizeof(float),
      index
    )
  );
  return 0;
}

#ifdef __cplusplus
}
#endif
