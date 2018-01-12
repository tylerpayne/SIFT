#include <chai.h>

namespace chai {
namespace cuda {
extern void cublas_safe_call(cublasStatus_t err);
extern void curand_safe_call(curandStatus_t err);

cudaStream_t _cudaStream;
cublasHandle_t _cublasHandle;
curandGenerator_t _curandGenerator;

void init_cuda_libs() {

  _cudaStream = 0;

  // cublas
  cublas_safe_call(cublasCreate(&_cublasHandle));

  // curand
  curand_safe_call(
      curandCreateGenerator(&_curandGenerator, CURAND_RNG_PSEUDO_DEFAULT));

  curand_safe_call(
      curandSetPseudoRandomGeneratorSeed(_curandGenerator, 1234ULL));
}
}
}
