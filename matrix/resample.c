#include <matrix.h>
#include <nppi.h>

#ifdef __cplusplus
extern "C" {
#endif

int resample(Matrix *a, Matrix *out, Shape shape)
{
  memassert(a,DEVICE);

  Matrix *ret = out;
  memassert(ret,DEVICE);

  Npp32f* pSrc = a->dev_ptr;
  int nSrcStep = a->shape.width*sizeof(float);
  NppiSize oSrcSize = {a->shape.width,a->shape.height};
  NppiRect oSrcROI = {0,0,a->shape.width,a->shape.height};

  Npp32f* pDst = ret->dev_ptr;
  int nDstStep = ret->shape.width*sizeof(float);
  NppiRect oDstROI = {0,0,shape.width,shape.height};

  double nXFactor = ((float)shape.width)/((float)a->shape.width);
  double nYFactor = ((float)shape.height)/((float)a->shape.height);
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
