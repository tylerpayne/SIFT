#include <chai.h>

namespace chai {
namespace cuda {
template <> __global__ void sqrt_kernel<float>(float *a, float *b, int len) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < len) {
    b[x] = sqrtf(a[x]);
  }
}

template <> __global__ void abs_kernel<float>(float *a, float *b, int len) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < len) {
    b[x] = fabsf(a[x]);
  }
}

template <> __global__ void exp_kernel<float>(float *a, float *b, int len) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < len) {
    b[x] = expf(a[x]);
  }
}

template <> __global__ void log_kernel<float>(float *a, float *b, int len) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < len) {
    b[x] = logf(a[x]);
  }
}

template <>
__global__ void pow_kernel<float>(float *a, float b, float *c, int len) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < len) {
    c[x] = powf(a[x], b);
  }
}
}
}
