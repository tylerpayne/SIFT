#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void range_reduce_kernel(float *a, float from, float to, int *d_index, Shape shape)
{
  extern __shared__ int results[];
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if (x<shape.width*shape.height)
  {
    if ((a[x]-to) < -0.01 && (a[x]-from) > -0.01)
    {
      results[threadIdx.x] = x;
    } else
    {
      results[threadIdx.x] = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      int counter = 1;
      for (int i = 0; i < blockDim.x; i++)
      {
        if (results[i] != 0)
          d_index[blockDim.x*blockIdx.x + counter++] = results[i];
      }
      d_index[blockDim.x*blockIdx.x] = counter;
    }
  }
}

#ifdef __cplusplus
}
#endif
