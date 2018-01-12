#include <chai.h>

namespace chai {
namespace cuda {
void cuda_safe_call(cudaError_t err) {
  if (err != cudaSuccess)
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
  // exit(1);
}

void cublas_safe_call(cublasStatus_t err) {
  if (err != CUBLAS_STATUS_SUCCESS)
    printf("CUBLAS ERROR: %i\n", err);
  // exit(1);
}

void curand_safe_call(curandStatus_t err) {
  if (err != CURAND_STATUS_SUCCESS)
    printf("CURAND ERROR: %i\n", err);
  // exit(1);
}

template <> void safe_call<cudaError_t>(cudaError_t err) {
  cuda_safe_call(err);
}

template <> void safe_call<cublasStatus_t>(cublasStatus_t err) {
  cublas_safe_call(err);
}

template <> void safe_call<curandStatus_t>(curandStatus_t err) {
  curand_safe_call(err);
}
}
}
