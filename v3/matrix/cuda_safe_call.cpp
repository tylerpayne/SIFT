#include <core.h>

void cuda_safe_call(cudaError_t err)
{
  if (err != cudaSuccess)
    printf("CUDA Error: %s\n",cudaGetErrorString(err));
    //exit(1);
}
