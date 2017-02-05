#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "ErrorTestKernel.cu"


int main(int argc, char const *argv[]) {

  int N = 128;

  size_t size = sizeof(int)*N;
  int* h_pI = (int*)malloc(size);
  int* h_pCount = (int*)malloc(sizeof(int));

  int* d_pI;
  int* d_pCount;
  cudaMalloc(&d_pI,size);
  cudaMalloc(&d_pCount,sizeof(int));

  dim3 bdim(min(1024,N));
  dim3 gdim(N/bdim.x + 1);
  printf("(%i,%i) (%i,%i)\n",bdim.x,bdim.y,gdim.x,gdim.y);
  atomicAddTest<<<gdim,bdim>>>(d_pI,d_pCount,N);
  cudaGetErrorString(cudaGetLastError());
  cudaMemcpy(h_pCount,d_pCount,sizeof(int),cudaMemcpyDeviceToHost);
  cudaFree(d_pI);
  cudaFree(d_pCount);

  printf("Count: %i\n",*h_pCount);

  free(h_pI);
  free(h_pCount);

  return 0;
}
