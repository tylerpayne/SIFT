#include <matrix.h>
#include <nppi.h>

#ifdef __cplusplus
extern "C" {
#endif

int convolve(Matrix* a, Matrix* b, Matrix* out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  memassert(ret,DEVICE);

  Npp32f* pSrc = a->dev_ptr;
  int nSrcStep = a->shape.width*sizeof(float);
  NppiSize oSrcSize = {a->shape.width,a->shape.height};
  NppiPoint oSrcOffset = {0,0};

  Npp32f* pDst = ret->dev_ptr;
  int nDstStep = a->shape.width*sizeof(float);
  NppiSize oSizeROI = {a->shape.width,a->shape.height};

  Npp32f* pKernel = b->dev_ptr;
  NppiSize oKernelSize = {b->shape.width,b->shape.height};
  NppiPoint oAnchor = {oKernelSize.width/2,oKernelSize.height/2};

  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

  npp_safe_call(nppiFilterBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,
                                      pDst,nDstStep,oSizeROI,
                                      pKernel,oKernelSize,oAnchor,
                                      eBorderType));

  return 0;
}

#ifdef __cplusplus
}
#endif
