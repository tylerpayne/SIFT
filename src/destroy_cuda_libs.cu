#include <chai.h>

namespace chai
{
  namespace cuda
  {
    extern void cublas_safe_call(cublasStatus_t err);
    extern void curand_safe_call(curandStatus_t err);

    extern cublasHandle_t _cublasHandle;
    extern curandGenerator_t _curandGenerator;

    void destroy_cuda_libs()
    {
      //cublas
      cublas_safe_call(cublasDestroy(_cublasHandle));

      //curand
      curand_safe_call(curandDestroyGenerator(_curandGenerator));
    }

  }
}
