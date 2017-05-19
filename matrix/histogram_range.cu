#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif
//ranges.shape.height = 1
//ranges.shape.width - 1 = length(d_histogram)
int histogram_range(Matrix *a, Matrix *ranges, int *d_histogram)
{
  memassert(a, DEVICE);
  memassert(ranges,DEVICE);
  Npp32f *pSrc = a->dev_ptr;
  int nSrcStep = a->shape.width*sizeof(float);
  NppiSize oSizeROI = {a->shape.width,a->shape.height};

  Npp32s *pHist = d_histogram;
  Npp32f *pLevels = ranges->dev_ptr;
  int nLevels = SHAPE2LEN(ranges->shape);
  Npp8u *pBuffer;

  int hpBufferSize;

  npp_safe_call(
     nppiHistogramRangeGetBufferSize_32f_C1R(
       oSizeROI, nLevels, &hpBufferSize
     )
  );

  cuda_safe_call(
    cudaMalloc((void **)&pBuffer,hpBufferSize)
  );

  npp_safe_call(
    nppiHistogramRange_32f_C1R (
      pSrc, nSrcStep, oSizeROI,
      pHist, pLevels, nLevels,
      pBuffer
    )
  );

  return 0;
}

#ifdef __cplusplus
}
#endif
