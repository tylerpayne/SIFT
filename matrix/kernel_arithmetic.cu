#include <chai.h>

namespace chai
{
  namespace cuda
  {
    extern __device__ int tuple_product(int_tuple shape);

    template<>
    __global__ void add_kernel<float>(float *a, float *b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = a[x] + b[x];
      }
    }

    template<>
    __global__ void addc_kernel<float>(float *a, float b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = a[x] + b;
      }
    }

    template<>
    __global__ void divide_kernel<float>(float *a, float *b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = a[x] / b[x];
      }
    }

    template<>
    __global__ void dividec_kernel<float>(float *a, float b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = a[x] / b;
      }
    }

    template<>
    __global__ void multiply_kernel<float>(float *a, float *b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = a[x] * b[x];
      }
    }

    template<>
    __global__ void multiplyc_kernel<float>(float *a, float b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = a[x] * b;
      }
    }

    template<>
    __global__ void subtract_kernel<float>(float *a, float *b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = a[x] - b[x];
      }
    }

    template<>
    __global__ void subtractc_kernel<float>(float *a, float b, float *c, int_tuple shape)
    {
      int x = blockDim.x*blockIdx.x + threadIdx.x;
      if (x<tuple_product(shape))
      {
        c[x] = a[x] - b;
      }
    }
  }
}
