#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t _cublasHandle;
extern curandGenerator_t _curandGenerator;

int destroy_cuda_libs()
{
  //cublas
  cublas_safe_call(cublasDestroy(_cublasHandle));

  //curand
  curand_safe_call(curandDestroyGenerator(_curandGenerator));

  return 0;
}


#ifdef __cplusplus
}
#endif
