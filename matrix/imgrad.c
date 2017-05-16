#include <matrix.h>
#include <nppi.h>

#ifdef __cplusplus
extern "C" {
#endif

int imgrad(Matrix* src, Matrix* dX, Matrix* dY, Matrix* mag, Matrix* angle)
{
  memassert(src,DEVICE);

  Npp32f* pSrc = src->dev_ptr;
  int nSrcStep = src->shape.height*sizeof(float);
  NppiSize oSrcSize = {src->shape.height,src->shape.width};
  NppiPoint oSrcOffset = {0,0};

  Npp32f* pDstX = NULL;
  Npp32f* pDstY = NULL;
  Npp32f* pDstMag = NULL;
  Npp32f* pDstAngle = NULL;

  int nDstXStep = 0;
  int nDstYStep = 0;
  int nDstMagStep = 0;
  int nDstAngleStep = 0;

  if (dX != NULL)
  {
    memassert(dX,DEVICE);
    pDstX = dX->dev_ptr;
    nDstXStep = nSrcStep;
  }

  if (dY != NULL)
  {
    memassert(dY,DEVICE);
    pDstY = dY->dev_ptr;
    nDstYStep = nSrcStep;
  }

  if (mag != NULL)
  {
    memassert(mag,DEVICE);
    pDstMag = mag->dev_ptr;
    nDstMagStep = nSrcStep;
  }

  if (angle != NULL)
  {
    memassert(angle,DEVICE);
    pDstAngle = angle->dev_ptr;
    nDstAngleStep = nSrcStep;
  }

  NppiSize oSizeROI = {src->shape.height,src->shape.width};
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

  npp_safe_call(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,
                                                    pDstX,nDstXStep,
                                                    pDstY,nDstYStep,
                                                    pDstMag,nDstMagStep,
                                                    pDstAngle,nDstAngleStep,
                                                    oSizeROI,eMaskSize,eNorm,eBorderType));
  return 0;
}

#ifdef __cplusplus
}
#endif
