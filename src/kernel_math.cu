#include <chai.h>

namespace chai {
  namespace cuda {
    extern __device__ int tuple_product(int_tuple shape);

    template<>
    __global__ void sqrt_kernel<float>(float *a, float *b, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x < tuple_product(shape))
      {
        b[x] = sqrtf(a[x]);
      }
    }

    template<>
    __global__ void abs_kernel<float>(float *a, float *b, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        b[x] = fabsf(a[x]);
      }
    }

    template<>
    __global__ void exp_kernel<float>(float *a, float *b, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        b[x] = expf(a[x]);
      }
    }

    template<>
    __global__ void log_kernel<float>(float *a, float *b, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        b[x] = logf(a[x]);
      }
    }

    template<>
    __global__ void pow_kernel<float>(float *a, float b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = powf(a[x],b);
      }
    }
  }
}
