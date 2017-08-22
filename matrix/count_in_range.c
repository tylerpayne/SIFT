#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;

int count_in_range(Matrix *a, float from, float to, int *count)
{
  memassert(a,DEVICE);

  Npp32f *pSrc = a->dev_ptr;
  int nSrcStep = a->shape.width*sizeof(float);
  NppiSize oSizeROI = {a->shape.width,a->shape.height};
  int *pCounts;
  cuda_safe_call(cudaMalloc((void **)&pCounts,sizeof(int)));
  Npp32f nLowerBound = from;
  Npp32f nUpperBound = to;

  Npp8u *pDeviceBuffer;
  int pBufferSize;

  npp_safe_call(
    nppiCountInRangeGetBufferHostSize_32f_C1R(
      oSizeROI, &pBufferSize
    )
  );

  cuda_safe_call(
    cudaMalloc((void **)&pDeviceBuffer,pBufferSize)
  );
  npp_safe_call(
     nppiCountInRange_32f_C1R (
       pSrc, nSrcStep, oSizeROI,
       pCounts, from, to,
       pDeviceBuffer
     )
  );

  cuda_safe_call(
    cudaFree((void *)pDeviceBuffer)
  );

  cuda_safe_call(
    cudaMemcpyAsync(
      count,pCounts,sizeof(int),
      cudaMemcpyDeviceToHost,_cudaStream
    )
  );

  return 0;
}


#ifdef __cplusplus
}
#endif
