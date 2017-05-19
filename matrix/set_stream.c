#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;
extern cublasHandle_t _cublasHandle;
extern curandGenerator_t _curandGenerator;

int set_stream(cudaStream_t stream)
{
  _cudaStream = stream;
  cublas_safe_call(
    cublasSetStream(_cublasHandle,stream)
  );
  curand_safe_call(
    curandSetStream(_curandGenerator,stream)
  );

  //nppSetStream(stream);
  return 0;
}

#ifdef __cplusplus
}
#endif
