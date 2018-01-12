
#include <chai.h>

namespace chai {
namespace cuda {
#define SET(K)                                                                 \
  template <> __global__ void add_kernel<K>(K * a, K * b, K * c, int len) {    \
    int x = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (x < len) {                                                             \
      c[x] = a[x] + b[x];                                                      \
    }                                                                          \
  }                                                                            \
  template <> __global__ void addc_kernel<K>(K * a, K b, K * c, int len) {     \
    int x = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (x < len) {                                                             \
      c[x] = a[x] + b;                                                         \
    }                                                                          \
  }                                                                            \
  template <> __global__ void divide_kernel<K>(K * a, K * b, K * c, int len) { \
    int x = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (x < len) {                                                             \
      c[x] = a[x] / b[x];                                                      \
    }                                                                          \
  }                                                                            \
  template <> __global__ void dividec_kernel<K>(K * a, K b, K * c, int len) {  \
    int x = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (x < len) {                                                             \
      c[x] = a[x] / b;                                                         \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  __global__ void multiply_kernel<K>(K * a, K * b, K * c, int len) {           \
    int x = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (x < len) {                                                             \
      c[x] = a[x] * b[x];                                                      \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  __global__ void multiplyc_kernel<K>(K * a, K b, K * c, int len) {            \
    int x = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (x < len) {                                                             \
      c[x] = a[x] * b;                                                         \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  __global__ void subtract_kernel<K>(K * a, K * b, K * c, int len) {           \
    int x = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (x < len) {                                                             \
      c[x] = a[x] - b[x];                                                      \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  __global__ void subtractc_kernel<K>(K * a, K b, K * c, int len) {            \
    int x = blockDim.x * blockIdx.x + threadIdx.x;                             \
    if (x < len) {                                                             \
      c[x] = a[x] - b;                                                         \
    }                                                                          \
  }

SET(char)
SET(int)
SET(float)
SET(double)
#undef SET
}
}
