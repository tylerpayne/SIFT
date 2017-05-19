#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaStream_t _cudaStream;
cublasHandle_t _cublasHandle;
curandGenerator_t _curandGenerator;

int init_cuda_libs()
{

  _cudaStream = 0;

  //cublas
  cublas_safe_call(
    cublasCreate(&_cublasHandle)
  );

  //curand
  curand_safe_call(
    curandCreateGenerator(
      &_curandGenerator, CURAND_RNG_PSEUDO_DEFAULT
    )
  );

  curand_safe_call(
    curandSetPseudoRandomGeneratorSeed(
      _curandGenerator, 1234ULL
    )
  );

  return 0;
}


#ifdef __cplusplus
}
#endif
