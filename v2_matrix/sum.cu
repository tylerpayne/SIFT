#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t _cublasHandle;

int sum(Matrix *a, float *out)
{
  memassert(a,DEVICE);
  cublas_safe_call(
    cublasSasum(_cublasHandle,
      SHAPE2LEN(a->shape),a->dev_ptr, 1,
      out
    )
  );
  return 0;
}

#ifdef __cplusplus
}
#endif
