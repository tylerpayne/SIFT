#include "MatrixUtil.h"
#include "ImageUtil.h"
#include <nppi.h>
#include "kernels/ImageKernels.cu"

#ifdef __cplusplus
  extern "C" {
#endif

void cudaErrCheck(cudaError_t stat)
{
  if (stat != cudaSuccess)
  {
    printf("CUDA ERR: %i\n",stat);
  }
}

void nppCallErrCheck(NppStatus status)
{
  if (status != NPP_SUCCESS)
  {
    printf("\n##########\nNPP ERROR!\nError code: %d\n##########\n",status);
  }
}

//#############
//INIT Methods
//############
void syncDeviceFromHostImpl(Image* self)
{
  copyHostToDeviceCudaMatrix(self->pixels);
}

void syncHostFromDeviceImpl(Image* self)
{
  copyDeviceToHostCudaMatrix(self->pixels);
}

void freeImageImpl(Image* self)
{
  self->pixels->free(self->pixels);
  free(self->pixbuf);
  free(self);
}

Image* newEmptyImageImpl(ImageUtil* self, int width, int height)
{
  if (VERBOSITY > 3)
  {
    printf("CREATING NEW EMPTY IMAGE\n");
  }
  Image* im = (Image*)malloc(sizeof(Image));
  im->nChannels = 1;
  int* shape = (int*)malloc(sizeof(int)*2);
  shape[0] = width;
  shape[1] = height;
  im->shape = shape;
  im->pixels=self->matutil->newEmptyMatrix(height,width);
  im->free = freeImageImpl;
  im->syncHostFromDevice = syncHostFromDeviceImpl;
  im->syncDeviceFromHost = syncDeviceFromHostImpl;
  return im;
}

Image* newImageImpl(ImageUtil* self, float* data, int width, int height)
{
  if (VERBOSITY > 3)
  {
    printf("CREATING NEW IMAGE FROM DATA\n");
  }
  Image* im = self->newEmptyImage(self,width,height);
  free(im->pixels->hostPtr);
  im->pixels->hostPtr = data;
  return im;
}

Image* newImageFromMatrixImpl(ImageUtil* self, Matrix* m)
{
  if (VERBOSITY > 3)
  {
    printf("CREATING NEW IMAGE FROM MATRIX\n");
  }
  Image* im = self->newEmptyImage(self,m->shape[1],m->shape[0]);
  im->pixels->free(im->pixels);
  im->pixels = m;
  return im;
}
//#################
//END INIT METHODS
//################

//#######
//FILTERS
//#######
Image* convolveImageImpl(ImageUtil* self, Image* im, Image* kernel)
{
  Image* retval = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (kernel->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(kernel->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape[0]*sizeof(float);
  NppiSize oSrcSize = {im->shape[0],im->shape[1]};
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDst = retval->pixels->devicePtr;
  int nDstStep = im->shape[0]*sizeof(float);
  NppiSize oSizeROI = {im->shape[0],im->shape[1]};
  Npp32f* pKernel = kernel->pixels->devicePtr;
  NppiSize oKernelSize = {kernel->shape[0],kernel->shape[1]};
  NppiPoint oAnchor = {oKernelSize.width/2,oKernelSize.height/2};
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiFilterBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDst,nDstStep,oSizeROI,pKernel,oKernelSize,oAnchor,eBorderType));

  return retval;
}

Image* gradientXImageImpl(ImageUtil* self, Image* im)
{
  Image* retval = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape[0]*sizeof(float);
  NppiSize oSrcSize = {im->shape[0],im->shape[1]};
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = retval->pixels->devicePtr;;
  int nDstXStep = retval->shape[0]*sizeof(float);;
  Npp32f* pDstY = NULL;
  int nDstYStep = 0;
  Npp32f* pDstMag = NULL;
  int nDstMagStep = 0;
  Npp32f* pDstAngle = NULL;
  int nDstAngleStep = 0;
  NppiSize oSizeROI = {im->shape[0],im->shape[1]};
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDstX,nDstXStep,pDstY,nDstYStep,pDstMag,nDstMagStep,pDstAngle,nDstAngleStep,oSizeROI,eMaskSize,eNorm,eBorderType));

  return retval;
}

Image* gradientYImageImpl(ImageUtil* self, Image* im)
{
  Image* retval = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape[0]*sizeof(float);
  NppiSize oSrcSize = {im->shape[0],im->shape[1]};
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = NULL;
  int nDstXStep = 0;
  Npp32f* pDstY = retval->pixels->devicePtr;;
  int nDstYStep = retval->shape[0]*sizeof(float);
  Npp32f* pDstMag = NULL;
  int nDstMagStep = 0;
  Npp32f* pDstAngle = NULL;
  int nDstAngleStep = 0;
  NppiSize oSizeROI = {im->shape[0],im->shape[1]};
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDstX,nDstXStep,pDstY,nDstYStep,pDstMag,nDstMagStep,pDstAngle,nDstAngleStep,oSizeROI,eMaskSize,eNorm,eBorderType));

  return retval;
}

Image* gradientMagnitudeImageImpl(ImageUtil* self, Image* im)
{
  Image* retval = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape[0]*sizeof(float);
  NppiSize oSrcSize = {im->shape[0],im->shape[1]};
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = NULL;
  int nDstXStep = 0;
  Npp32f* pDstY = NULL;
  int nDstYStep = 0;
  Npp32f* pDstMag = retval->pixels->devicePtr;
  int nDstMagStep = retval->shape[0]*sizeof(float);
  Npp32f* pDstAngle = NULL;
  int nDstAngleStep = 0;
  NppiSize oSizeROI = {im->shape[0],im->shape[1]};
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDstX,nDstXStep,pDstY,nDstYStep,pDstMag,nDstMagStep,pDstAngle,nDstAngleStep,oSizeROI,eMaskSize,eNorm,eBorderType));

  return retval;
}

Image* gradientAngleImageImpl(ImageUtil* self, Image* im)
{
  Image* retval = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape[0]*sizeof(float);
  NppiSize oSrcSize = {im->shape[0],im->shape[1]};
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = NULL;
  int nDstXStep = 0;
  Npp32f* pDstY = NULL;
  int nDstYStep = 0;
  Npp32f* pDstMag = NULL;
  int nDstMagStep = 0;
  Npp32f* pDstAngle = retval->pixels->devicePtr;
  int nDstAngleStep = retval->shape[0]*sizeof(float);
  NppiSize oSizeROI = {im->shape[0],im->shape[1]};
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDstX,nDstXStep,pDstY,nDstYStep,pDstMag,nDstMagStep,pDstAngle,nDstAngleStep,oSizeROI,eMaskSize,eNorm,eBorderType));

  return retval;
}

ImageGradientVectorPair* gradientsImageImpl(ImageUtil* self, Image* im)
{
  Image* magnitude = self->newEmptyImage(self,im->shape[0],im->shape[1]);
  Image* angle = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (magnitude->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(magnitude->pixels);
  }
  if (angle->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(angle->pixels);
  }

  ImageGradientVectorPair* retval = (ImageGradientVectorPair*)malloc(sizeof(ImageGradientVectorPair));
  retval->magnitude = magnitude;
  retval->angle = angle;
  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape[0]*sizeof(float);
  NppiSize oSrcSize = {im->shape[0],im->shape[1]};
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = NULL;
  int nDstXStep = 0;
  Npp32f* pDstY = NULL;
  int nDstYStep = 0;
  Npp32f* pDstMag = magnitude->pixels->devicePtr;
  int nDstMagStep = magnitude->shape[0]*sizeof(float);
  Npp32f* pDstAngle = angle->pixels->devicePtr;
  int nDstAngleStep = angle->shape[0]*sizeof(float);
  NppiSize oSizeROI = {im->shape[0],im->shape[1]};
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDstX,nDstXStep,pDstY,nDstYStep,pDstMag,nDstMagStep,pDstAngle,nDstAngleStep,oSizeROI,eMaskSize,eNorm,eBorderType));

  return retval;
}
//############
//END FILTERS
//###########

//#########
//GEOMETRY
//########
Image* resampleImageImpl(ImageUtil* self, Image* im, int w, int h)
{
  Image* retval = self->newEmptyImage(self,w,h);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape[0]*sizeof(float);
  NppiSize oSrcSize = {im->shape[0],im->shape[1]};
  NppiRect oSrcROI = {0,0,im->shape[0],im->shape[1]};
  Npp32f* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape[0]*sizeof(float);
  NppiRect oDstROI = {0,0,retval->shape[0],retval->shape[1]};
  double nXFactor = ((float)w)/((float)im->shape[0]);
  double nYFactor = ((float)h)/((float)im->shape[1]);
  double nXShift = 0;
  double nYShift = 0;
  NppiInterpolationMode eInterpolation = NPPI_INTER_CUBIC;
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiResizeSqrPixel_32f_C1R(pSrc,oSrcSize,nSrcStep,oSrcROI,pDst,nDstStep,oDstROI,nXFactor,nYFactor,nXShift,nYShift,eInterpolation));

  return retval;
}
//#############
//END GEOMETRY
//#############

//##########
//ARITHMETIC
//##########
Image* addImageImpl(ImageUtil* self, Image* im1, Image*im2)
{
  Image* retval = self->newEmptyImage(self,im1->shape[0],im1->shape[1]);

  if (im1->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im1->pixels);
  }
  if (im2->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im2->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  Npp32f* pSrc1 = im1->pixels->devicePtr;
  int nSrc1Step = im1->shape[0]*sizeof(float);
  Npp32f* pSrc2 = im2->pixels->devicePtr;
  int nSrc2Step = im2->shape[0]*sizeof(float);
  Npp32f* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape[0]*sizeof(float);
  NppiSize oSizeROI = {im1->shape[0],im1->shape[1]};
//  nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiAdd_32f_C1R(pSrc1,nSrc1Step,pSrc2,nSrc2Step,pDst,nDstStep,oSizeROI));

  return retval;
}

Image* subtractImageImpl(ImageUtil* self, Image* im1, Image*im2)
{
  Image* retval = self->newEmptyImage(self,im1->shape[0],im1->shape[1]);

  if (im1->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im1->pixels);
  }
  if (im2->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im2->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  Npp32f* pSrc1 = (Npp32f*)(im1->pixels->devicePtr);
  int nSrc1Step = im1->shape[0]*sizeof(float);
  Npp32f* pSrc2 = im2->pixels->devicePtr;
  int nSrc2Step = im2->shape[0]*sizeof(float);
  Npp32f* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape[0]*sizeof(float);
  NppiSize oSizeROI = {im1->shape[0],im1->shape[1]};
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiSub_32f_C1R(pSrc1,nSrc1Step,pSrc2,nSrc2Step,pDst,nDstStep,oSizeROI));

  return retval;
}

Image* multiplyCImageImpl(ImageUtil* self, Image* im, float val)
{
  Image* retval = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  float* pSrc1 = im->pixels->devicePtr;
  int nSrc1Step = im->shape[0]*sizeof(float);
  float nConstant = val;
  float* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape[0]*sizeof(float);
  NppiSize oSizeROI = {im->shape[0],im->shape[1]};
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiMulC_32f_C1R(pSrc1,nSrc1Step,nConstant,pDst,nDstStep,oSizeROI));

  return retval;
}

Image* multiplyImageImpl(ImageUtil* self, Image* im1, Image* im2)
{
  Image* retval = self->newEmptyImage(self,im1->shape[0],im1->shape[1]);

  if (im1->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im1->pixels);
  }
  if (im2->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im2->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }


  float* pSrc1 = im1->pixels->devicePtr;
  int nSrc1Step = im1->shape[0]*sizeof(float);
  float* pSrc2 = im2->pixels->devicePtr;
  int nSrc2Step = im2->shape[0]*sizeof(float);
  float* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape[0]*sizeof(float);
  NppiSize oSizeROI = {im1->shape[0],im1->shape[1]};
  //nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiMul_32f_C1R(pSrc1,nSrc1Step,pSrc2,nSrc2Step,pDst,nDstStep,oSizeROI));

  return retval;
}

//###############
//END ARITHMETIC
//##############

//##########
//STATISTICS
//##########
Image* maxImageImpl(ImageUtil* self, Image* im, int radius)
{
  Image* retval = self->newEmptyImage(self,im->shape[0],im->shape[1]);
  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  int wregions = im->shape[0]/radius;
  int hregions = im->shape[1]/radius;
  float* pSrc = im->pixels->devicePtr;
  float* pDst = retval->pixels->devicePtr;
  NppiSize oSize = {im->shape[0],im->shape[1]};

  int bdimX = fmin(32,wregions);
  int bdimY = fmin(32,hregions);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(wregions/bdimX + 1,hregions/bdimY + 1);
  //printf("w: %i, h: %i, bdimX: %i, bdimY: %i, gdimX: %i, gdimY: %i",wregions,hregions,bdimX,bdimY,gdim.x,gdim.y);
  LocalMaxKernel<<<gdim,bdim>>>(pSrc,pDst,oSize,radius);

  return retval;
}

ImageIndexPair* maxIdxImageImpl(ImageUtil* self, Image* im, int radius)
{
  Image* dst = self->newEmptyImage(self,im->shape[0],im->shape[1]);
  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (dst->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(dst->pixels);
  }
  int wregions = im->shape[0]/radius;
  int hregions = im->shape[1]/radius;
  Npp32f* pSrc = im->pixels->devicePtr;
  Npp32f* pDst = dst->pixels->devicePtr;
  int* pIdx;
  size_t indexSize = sizeof(int)*(wregions*hregions);
  cudaErrCheck(cudaMalloc(&pIdx,indexSize));
  NppiSize oSize = {im->shape[0],im->shape[1]};

  int bdimX = fmin(32,wregions);
  int bdimY = fmin(32,hregions);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(wregions/bdimX + 1,hregions/bdimY + 1);
  LocalMaxIdxKernel<<<gdim,bdim>>>(pSrc,pDst,pIdx,oSize,radius);
  int* h_pIdx = (int*)malloc(indexSize);
  cudaErrCheck(cudaMemcpy(h_pIdx,pIdx,indexSize,cudaMemcpyDeviceToHost));
  ImageIndexPair* retval = (ImageIndexPair*)malloc(sizeof(ImageIndexPair));
  retval->image=im;
  retval->index=h_pIdx;
  retval->count=wregions*hregions;
  retval->subPixelX = NULL;
  retval->subPixelY = NULL;
  cudaErrCheck(cudaFree(pIdx));
  return retval;
}

ImageIndexPair* thresholdIdxImageImpl(ImageUtil* self, Image* im, float threshold)
{
  Image* dst = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (dst->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(dst->pixels);
  }

  Npp32f* pSrc = im->pixels->devicePtr;
  Npp32f* pDst = dst->pixels->devicePtr;
  int* pIdx;
  size_t indexSize = sizeof(int)*(im->shape[0]*im->shape[1]);
  cudaErrCheck(cudaMalloc(&pIdx,indexSize));

  int bdimX = fmin(1024,im->shape[0]*im->shape[1]);
  dim3 bdim(bdimX);
  dim3 gdim(im->shape[0]*im->shape[1]/bdimX + 1);
  ThresholdIdxKernel<<<gdim,bdim>>>(pSrc,pDst,threshold,im->shape[0]*im->shape[1],pIdx);

  int* h_pIdx = (int*)malloc(indexSize);
  cudaErrCheck(cudaMemcpy(h_pIdx,pIdx,indexSize,cudaMemcpyDeviceToHost));
  int count = 0;
  for (int i = 0; i < im->shape[0]*im->shape[1]; i++)
  {
    //printf("? %i\t",h_pIdx[i]);
    if (h_pIdx[i] == 0)
    {
      continue;
    }
    if (h_pIdx[i] < 0 || h_pIdx[i] >= im->shape[0]*im->shape[1])
    {
      continue;
    }
    count++;
  }
  int* arrayIndex = (int*)malloc(sizeof(int)*count);
  int c = 0;
  for (int i = 0; i < im->shape[0]*im->shape[1]; i++)
  {
    if (h_pIdx[i] == 0)
    {
      continue;
    }
    if (h_pIdx[i] < 0 || h_pIdx[i] >= im->shape[0]*im->shape[1])
    {
      continue;
    }
    arrayIndex[c] = h_pIdx[i];
    c++;
  }
  ImageIndexPair* retval = (ImageIndexPair*)malloc(sizeof(ImageIndexPair));
  retval->image=dst;
  retval->index=arrayIndex;
  retval->count=count;
  free(h_pIdx);
  return retval;
}

Image* localContrastImageImpl(ImageUtil* self, Image* im, int radius)
{
  Image* retval = self->newEmptyImage(self,im->shape[0],im->shape[1]);

  if (im->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(im->pixels);
  }
  if (retval->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(retval->pixels);
  }

  int wregions = im->shape[0]/radius;
  int hregions = im->shape[1]/radius;
  float* pSrc = im->pixels->devicePtr;
  float* pDst = retval->pixels->devicePtr;
  NppiSize oSize = {im->shape[0],im->shape[1]};

  int bdimX = fmin(32,wregions);
  int bdimY = fmin(32,hregions);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(wregions/bdimX + 1,hregions/bdimY + 1);
  //printf("w: %i, h: %i, bdimX: %i, bdimY: %i, gdimX: %i, gdimY: %i",wregions,hregions,bdimX,bdimY,gdim.x,gdim.y);
  LocalContrastKernel<<<gdim,bdim>>>(pSrc,pDst,oSize,radius);

  return retval;
}
//##############
//END STATISTICS
//##############

//######################
// BEGIN COMPUTERVISION
//######################

void subPixelAlignImageIndexPairImpl(ImageUtil* self, ImageIndexPair* data)
{
  if (data->image->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(data->image->pixels);
  }
  Image* Ix = self->gradientX(self,data->image);
  Image* Iy = self->gradientY(self,data->image);
  Image* Ixx = self->gradientX(self,Ix);
  Image* Ixy = self->gradientY(self,Ix);
  Image* Iyy = self->gradientY(self,Iy);

  float* pSubPixelX;
  float* pSubPixelY;
  NppiSize oSize = {Ix->shape[0],Ix->shape[1]};

  int* d_pIdx;
  cudaErrCheck(cudaMalloc(&d_pIdx,sizeof(int)*data->count));
  cudaErrCheck(cudaMemcpy(d_pIdx,data->index,sizeof(int)*data->count,cudaMemcpyHostToDevice));

  int size = sizeof(float)*data->count;
  cudaErrCheck(cudaMalloc(&pSubPixelX,size));
  cudaErrCheck(cudaMalloc(&pSubPixelY,size));

  int bdimX = min(1024,data->count);
  dim3 bdim(bdimX);
  dim3 gdim((data->count/bdimX) + 1);
  SubPixelAlignKernel<<<gdim,bdim>>>(Ix->pixels->devicePtr,Iy->pixels->devicePtr,Ixx->pixels->devicePtr,Ixy->pixels->devicePtr,Iyy->pixels->devicePtr,d_pIdx,pSubPixelX,pSubPixelY,oSize,data->count);
  data->subPixelX = (float*)malloc(size);
  data->subPixelY = (float*)malloc(size);

  cudaErrCheck(cudaMemcpy(data->subPixelX,pSubPixelX,size,cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(data->subPixelY,pSubPixelY,size,cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaFree(pSubPixelX));
  cudaErrCheck(cudaFree(pSubPixelY));
  cudaErrCheck(cudaFree(d_pIdx));
  Ix->free(Ix);
  Iy->free(Iy);
}


Matrix** makeFeatureDescriptorsForImageIndexPairImpl(ImageUtil* self, ImageIndexPair* keypoints, Image* im, int featureWidth)
{
  float* d_features;
  float* d_subPixelX;
  float* d_subPixelY;
  cudaErrCheck(cudaMalloc(&d_features,sizeof(float)*featureWidth*featureWidth*keypoints->count));
  cudaErrCheck(cudaMalloc(&d_subPixelY,sizeof(float)*keypoints->count));
  cudaErrCheck(cudaMalloc(&d_subPixelX,sizeof(float)*keypoints->count));
  cudaMemcpy(d_subPixelX,keypoints->subPixelX,sizeof(float)*keypoints->count,cudaMemcpyHostToDevice);
  cudaMemcpy(d_subPixelY,keypoints->subPixelY,sizeof(float)*keypoints->count,cudaMemcpyHostToDevice);

  NppiSize oSize = {im->shape[0],im->shape[1]};

  int bdimX = min(1024,keypoints->count);
  dim3 bdim(bdimX);
  dim3 gdim((keypoints->count/bdimX) + 1);
  MakeFeatureDescriptorKernel<<<gdim, bdim>>>(im->pixels->devicePtr,oSize,d_subPixelX,d_subPixelY,keypoints->count,d_features,featureWidth);
  Matrix** retval = (Matrix**)malloc(sizeof(Matrix*)*keypoints->count);
  for (int i = 0; i < keypoints->count; i++)
  {
    float* h_feature = (float*)malloc(sizeof(float)*featureWidth*featureWidth);
    cudaErrCheck(cudaMemcpy(h_feature,&d_features[i*featureWidth*featureWidth],sizeof(float)*featureWidth*featureWidth,cudaMemcpyDeviceToHost));
    Matrix* m = self->matutil->newMatrix(h_feature,featureWidth,featureWidth);
    retval[i] = m;
  }
  cudaErrCheck(cudaFree(d_subPixelX));
  cudaErrCheck(cudaFree(d_subPixelY));
  cudaErrCheck(cudaFree(d_features));
  return retval;
}

Matrix* generalizeFeatureMatrixImpl(ImageUtil* self, Matrix* features, int nBins)
{
  if (features->isHostSide)
  {
    copyHostToDeviceCudaMatrix(features);
  }
  int nFeatureWidth = (int)sqrt(features->shape[1]);
  Matrix* genFeatures = self->matutil->newEmptyMatrix(features->shape[0],nBins*nFeatureWidth);
  copyHostToDeviceCudaMatrix(genFeatures);
  int bdimX = 16;
  int bdimY = fmin(64,features->shape[0]);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(1,features->shape[0]/bdimY + 1);
  GeneralizeFeatureKernel<<<gdim,bdim>>>(features->devicePtr,genFeatures->devicePtr,features->shape[0],features->shape[1],16,nBins);
  return genFeatures;
}

void unorientFeatureMatrixImpl(ImageUtil* self, Matrix* features, int nBins)
{
  if (features->isHostSide)
  {
    copyHostToDeviceCudaMatrix(features);
  }
  int bdimX = fmin(1024,features->shape[0]);
  dim3 bdim(bdimX);
  dim3 gdim(features->shape[0]/bdimX + 1);
  UnorientFeatureKernel<<<gdim,bdim,sizeof(float)*nBins>>>(features->devicePtr,features->shape[0],features->shape[1],nBins);
}

//########
// END CV
//########


DLLEXPORT ImageUtil* GetImageUtil(MatrixUtil* matutil)
{
  ImageUtil* self = (ImageUtil*)malloc(sizeof(ImageUtil));

  self->matutil = matutil;
  self->newEmptyImage = newEmptyImageImpl;
  self->newImage = newImageImpl;
  self->newImageFromMatrix = newImageFromMatrixImpl;
  self->resample = resampleImageImpl;
  self->add = addImageImpl;
  self->subtract = subtractImageImpl;
  self->multiply = multiplyImageImpl;
  self->multiplyC = multiplyCImageImpl;
  self->max = maxImageImpl;
  self->maxIdx = maxIdxImageImpl;
  self->localContrast = localContrastImageImpl;
  self->thresholdIdx = thresholdIdxImageImpl;
  self->gradientX = gradientXImageImpl;
  self->gradientY = gradientYImageImpl;
  self->gradientMagnitude = gradientMagnitudeImageImpl;
  self->gradientAngle = gradientAngleImageImpl;
  self->gradients = gradientsImageImpl;
  self->convolve = convolveImageImpl;
  self->subPixelAlignImageIndexPair = subPixelAlignImageIndexPairImpl;
  self->makeFeatureDescriptorsForImageIndexPair = makeFeatureDescriptorsForImageIndexPairImpl;
  self->unorientFeatureMatrix = unorientFeatureMatrixImpl;
  self->generalizeFeatureMatrix = generalizeFeatureMatrixImpl;

  return self;
}

#ifdef __cplusplus
  }
#endif
