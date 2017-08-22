#include <matrix.h>
#include <nppi.h>

#ifdef __cplusplus
extern "C" {
#endif

int dilate(Matrix *a, unsigned char *d_b, Shape b_shape, Matrix *out)
{
  memassert(a,DEVICE);

  Matrix *ret = out;
  memassert(ret,DEVICE);

  Npp32f* pSrc = a->dev_ptr;
  int nSrcStep = a->shape.width*sizeof(float);
  NppiSize oSrcSize = {a->shape.width,a->shape.height};
  NppiPoint oSrcOffset = {0,0};

  Npp32f* pDst = ret->dev_ptr;
  int nDstStep = ret->shape.width*sizeof(float);
  NppiSize oSizeROI = {b_shape.width,b_shape.height};

  Npp8u *pMask = d_b;
  NppiSize oMaskSize = {b_shape.width,b_shape.height};
  NppiPoint oAnchor = {b_shape.width/2,b_shape.height/2};

  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

  npp_safe_call(
    nppiDilateBorder_32f_C1R (
      pSrc,nSrcStep,oSrcSize,oSrcOffset,
      pDst,nDstStep,oSizeROI,
      pMask,oMaskSize,oAnchor,
      eBorderType
    )
  );

  return 0;
}

#ifdef __cplusplus
}
#endif
