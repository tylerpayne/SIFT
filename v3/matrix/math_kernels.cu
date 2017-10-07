#include <matrix.h>

template<typename K>
__global__ void sqrt_kernel(K *a, K *b, Tuple<int> shape);

template<typename K>
__global__ void abs_kernel(K *a, K *b, Tuple<int> shape);

template<typename K>
__global__ void exp_kernel(K *a, K *b, Tuple<int> shape);

template<typename K>
__global__ void log_kernel(K *a, K *b, Tuple<int> shape);

template<typename K>
__global__ void pow_kernel(K *a, K b, K *c, Tuple<int> shape);

template<>
__global__ void sqrt_kernel<float>(float *a, float *b, Tuple<int> shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x < shape.product())
  {
    b[x] = sqrtf(a[x]);
  }
}

template<>
__global__ void abs_kernel<float>(float *a, float *b, Tuple<int> shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.product())
  {
    b[x] = fabsf(a[x]);
  }
}

template<>
__global__ void exp_kernel<float>(float *a, float *b, Tuple<int> shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.product())
  {
    b[x] = expf(a[x]);
  }
}

template<>
__global__ void log_kernel<float>(float *a, float *b, Tuple<int> shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.product())
  {
    b[x] = logf(a[x]);
  }
}

template<>
__global__ void pow_kernel<float>(float *a, float b, float *c, Tuple<int> shape)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.product())
  {
    c[x] = powf(a[x],b);
  }
}
