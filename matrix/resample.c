#include <matrix.h>
#include <nppi.h>

#ifdef __cplusplus
extern "C" {
#endif

int resample(Matrix* a, Matrix* out, Shape shape)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Npp32f* pSrc = a->dev_ptr;
  int nSrcStep = a->shape.height*sizeof(float);
  NppiSize oSrcSize = {a->shape.height,a->shape.width};
  NppiRect oSrcROI = {0,0,a->shape.height,a->shape.width};

  Npp32f* pDst = out->dev_ptr;
  int nDstStep = out->shape.height*sizeof(float);
  NppiRect oDstROI = {0,0,out->shape.height,out->shape.width};

  double nXFactor = ((float)shape.width)/((float)a->shape.height);
  double nYFactor = ((float)shape.height)/((float)a->shape.width);
  double nXShift = 0;
  double nYShift = 0;
  NppiInterpolationMode eInterpolation = NPPI_INTER_CUBIC;

  npp_safe_call(nppiResizeSqrPixel_32f_C1R(pSrc,oSrcSize,nSrcStep,oSrcROI,
                                        pDst,nDstStep,oDstROI,
                                        nXFactor,nYFactor,
                                        nXShift,nYShift,
                                        eInterpolation));

  return 0;
}

#ifdef __cplusplus
}
#endif
