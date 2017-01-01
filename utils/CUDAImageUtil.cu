#include "CUDAMatrixUtil.cu"
#include "ImageUtil.h"

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
Image* newEmptyImageImpl(ImageUtil* self, int width, int height)
{
  if (VERBOSITY > 3)
  {
    printf("CREATING NEW EMPTY IMAGE\n");
  }
  Image* im = (Image*)malloc(sizeof(Image));
  im->nChannels = 1;
  NppiSize shape = {width,height};
  im->shape = shape;
  im->pixels=self->matutil->newEmptyMatrix(width,height);
  copyHostToDeviceCudaMatrix(self->matutil,im->pixels);
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
  copyHostToDeviceCudaMatrix(self->matutil,im->pixels);
  return im;
}
//#################
//END INIT METHODS
//################

//#############
//FILE HANDLING
//#############
Image* loadImageFromFileImpl(ImageUtil* self, char* p)
{
  unsigned error;
  unsigned char* image;
  unsigned width, height;
  const char* path = p;
  error = lodepng_decode32_file(&image, &height, &width, path);
  if (error)
  {
    printf("error %u: %s\n", error, lodepng_error_text(error));
  }
  int w = (int)width;
  int h = (int)height;
  int size = sizeof(float)*w*h;
  float* data = (float*)malloc(size);
  int counter = 0;
  for (int i = 0; i < w*h*4; i+=4)
  {
    data[counter] = 0.21*((float)(int)image[i])/255.0;
    data[counter] += 0.72*((float)(int)image[i+1])/255.0;
    data[counter] += 0.07*((float)(int)image[i+2])/255.0;
    counter++;
  }
  Image* im = self->newImage(self,data,w,h);
  return im;
}

void saveImageToFileImpl(ImageUtil* self, Image* im, char* p)
{
  /*Encode the image*/
  if (VERBOSITY > 0)
  {
      printf("###########    Saving File: ' %s '   ############",p);
  }
  cudaDeviceSynchronize();
  if (!im->pixels->isHostSide)
  {
    copyDeviceToHostCudaMatrix(self->matutil,im->pixels);
  }
  cudaDeviceSynchronize();
  float* pix = im->pixels->hostPtr;
  const char* path = p;
  unsigned char* saveim = (unsigned char*)malloc(sizeof(unsigned char)*im->shape.width*im->shape.height*4);
  int pixelcount = 0;

  float mmm = 0.0;
  for (int k=0; k<(im->shape.width * im->shape.height); k++)
  {
    if (fabsf(pix[k])> mmm)
    {
      mmm = fabsf(pix[k]);
    }
  }
  if (mmm != 1.0)
  {
    for (int j=0; j<(im->shape.width * im->shape.height); j++)
    {
      //printf("[ %f ]",pix[j]);
      pix[j] =fabsf(pix[j])/mmm;
    }
  }
  if (VERBOSITY > 1)
  {
    printf("\nSaved Image Pixel Maximum: %f\n",mmm);
  }
  for (int i = 0; i < im->shape.width*im->shape.height*4; i+=4)
  {
    saveim[i] = pix[pixelcount] * 255.0;
    saveim[i+1] = pix[pixelcount] * 255.0;;
    saveim[i+2] = pix[pixelcount] * 255.0;;
    saveim[i+3] = 255;
    pixelcount++;
  }

  unsigned error = lodepng_encode32_file(path, saveim, im->shape.height, im->shape.width);

  /*if there's an error, display it*/
  if (error)
  {
    printf("error %u: %s\n", error, lodepng_error_text(error));
  }
}

//##################
//END FILE HANDLING
//#################

//#######
//FILTERS
//#######
Image* convolveImageImpl(ImageUtil* self, Image* im, Image* kernel)
{
  Image* retval = self->newEmptyImage(self,im->shape.width,im->shape.height);

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape.width*sizeof(float);
  NppiSize oSrcSize = im->shape;
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDst = retval->pixels->devicePtr;
  int nDstStep = im->shape.width*sizeof(float);
  NppiSize oSizeROI = im->shape;
  Npp32f* pKernel = kernel->pixels->devicePtr;
  NppiSize oKernelSize = kernel->shape;
  NppiPoint oAnchor = {oKernelSize.width/2,oKernelSize.height/2};
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiFilterBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDst,nDstStep,oSizeROI,pKernel,oKernelSize,oAnchor,eBorderType));

  return retval;
}


Image* gradientMagnitudeImageImpl(ImageUtil* self, Image* im, NppiMaskSize eMaskSize)
{
  Image* retval = self->newEmptyImage(self,im->shape.width,im->shape.height);

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape.width*sizeof(float);
  NppiSize oSrcSize = im->shape;
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = NULL;
  int nDstXStep = 0;
  Npp32f* pDstY = NULL;
  int nDstYStep = 0;
  Npp32f* pDstMag = retval->pixels->devicePtr;
  int nDstMagStep = retval->shape.width*sizeof(float);
  Npp32f* pDstAngle = NULL;
  int nDstAngleStep = 0;
  NppiSize oSizeROI = im->shape;
  NppiNorm eNorm = nppiNormL2;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDstX,nDstXStep,pDstY,nDstYStep,pDstMag,nDstMagStep,pDstAngle,nDstAngleStep,oSizeROI,eMaskSize,eNorm,eBorderType));

  return retval;
}

Image* gradientMagnitudeImageImpl(ImageUtil* self, Image* im)
{
  Image* retval = self->newEmptyImage(self,im->shape.width,im->shape.height);

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape.width*sizeof(float);
  NppiSize oSrcSize = im->shape;
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = NULL;
  int nDstXStep = 0;
  Npp32f* pDstY = NULL;
  int nDstYStep = 0;
  Npp32f* pDstMag = retval->pixels->devicePtr;
  int nDstMagStep = retval->shape.width*sizeof(float);
  Npp32f* pDstAngle = NULL;
  int nDstAngleStep = 0;
  NppiSize oSizeROI = im->shape;
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDstX,nDstXStep,pDstY,nDstYStep,pDstMag,nDstMagStep,pDstAngle,nDstAngleStep,oSizeROI,eMaskSize,eNorm,eBorderType));

  return retval;
}

Image* gradientAngleImageImpl(ImageUtil* self, Image* im)
{
  Image* retval = self->newEmptyImage(self,im->shape.width,im->shape.height);

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape.width*sizeof(float);
  NppiSize oSrcSize = im->shape;
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = NULL;
  int nDstXStep = 0;
  Npp32f* pDstY = NULL;
  int nDstYStep = 0;
  Npp32f* pDstMag = NULL;
  int nDstMagStep = 0;
  Npp32f* pDstAngle = retval->pixels->devicePtr;
  int nDstAngleStep = retval->shape.width*sizeof(float);
  NppiSize oSizeROI = im->shape;
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,pDstX,nDstXStep,pDstY,nDstYStep,pDstMag,nDstMagStep,pDstAngle,nDstAngleStep,oSizeROI,eMaskSize,eNorm,eBorderType));

  return retval;
}

ImageGradientVectorPair gradientsImageImpl(ImageUtil* self, Image* im)
{
  Image* magnitude = self->newEmptyImage(self,im->shape.width,im->shape.height);
  Image* angle = self->newEmptyImage(self,im->shape.width,im->shape.height);
  ImageGradientVectorPair retval = {magnitude,angle};
  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape.width*sizeof(float);
  NppiSize oSrcSize = im->shape;
  NppiPoint oSrcOffset = {0,0};
  Npp32f* pDstX = NULL;
  int nDstXStep = 0;
  Npp32f* pDstY = NULL;
  int nDstYStep = 0;
  Npp32f* pDstMag = magnitude->pixels->devicePtr;
  int nDstMagStep = magnitude->shape.width*sizeof(float);
  Npp32f* pDstAngle = angle->pixels->devicePtr;
  int nDstAngleStep = angle->shape.width*sizeof(float);
  NppiSize oSizeROI = im->shape;
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;
  nppSetStream(self->matutil->stream);
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

  Npp32f* pSrc = im->pixels->devicePtr;
  int nSrcStep = im->shape.width*sizeof(float);
  NppiSize oSrcSize = im->shape;
  NppiRect oSrcROI = {0,0,im->shape.width,im->shape.height};
  Npp32f* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape.width*sizeof(float);
  NppiRect oDstROI = {0,0,retval->shape.width,retval->shape.height};
  double nXFactor = ((float)w)/((float)im->shape.width);
  double nYFactor = ((float)h)/((float)im->shape.height);
  double nXShift = 0;
  double nYShift = 0;
  NppiInterpolationMode eInterpolation = NPPI_INTER_CUBIC;
  nppSetStream(self->matutil->stream);
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
  Image* retval = self->newEmptyImage(self,im1->shape.width,im1->shape.height);

  Npp32f* pSrc1 = im1->pixels->devicePtr;
  int nSrc1Step = im1->shape.width*sizeof(float);
  Npp32f* pSrc2 = im2->pixels->devicePtr;
  int nSrc2Step = im2->shape.width*sizeof(float);
  Npp32f* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape.width*sizeof(float);
  NppiSize oSizeROI = im1->shape;
  nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiAdd_32f_C1R(pSrc1,nSrc1Step,pSrc2,nSrc2Step,pDst,nDstStep,oSizeROI));

  return retval;
}

Image* subtractImageImpl(ImageUtil* self, Image* im1, Image*im2)
{
  Image* retval = self->newEmptyImage(self,im1->shape.width,im1->shape.height);

  Npp32f* pSrc1 = im1->pixels->devicePtr;
  int nSrc1Step = im1->shape.width*sizeof(float);
  Npp32f* pSrc2 = im2->pixels->devicePtr;
  int nSrc2Step = im2->shape.width*sizeof(float);
  Npp32f* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape.width*sizeof(float);
  NppiSize oSizeROI = im1->shape;
  nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiSub_32f_C1R(pSrc1,nSrc1Step,pSrc2,nSrc2Step,pDst,nDstStep,oSizeROI));

  return retval;
}

Image* multiplyCImageImpl(ImageUtil* self, Image* im, float val)
{
  Image* retval = self->newEmptyImage(self,im->shape.width,im->shape.height);

  float* pSrc1 = im->pixels->devicePtr;
  int nSrc1Step = im->shape.width*sizeof(float);
  float nConstant = val;
  float* pDst = retval->pixels->devicePtr;
  int nDstStep = retval->shape.width*sizeof(float);
  NppiSize oSizeROI = im->shape;
  nppSetStream(self->matutil->stream);
  nppCallErrCheck(nppiMulC_32f_C1R(pSrc1,nSrc1Step,nConstant,pDst,nDstStep,oSizeROI));

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
  Image* retval = self->newEmptyImage(self,im->shape.width,im->shape.height);

  int wregions = im->shape.width/radius;
  int hregions = im->shape.height/radius;
  float* pSrc = im->pixels->devicePtr;
  float* pDst = retval->pixels->devicePtr;
  NppiSize oSize = im->shape;

  int bdimX = fmin(32,wregions);
  int bdimY = fmin(32,hregions);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(wregions/bdimX,hregions/bdimY);
  printf("w: %i, h: %i, bdimX: %i, bdimY: %i, gdimX: %i, gdimY: %i",wregions,hregions,bdimX,bdimY,gdim.x,gdim.y);
  LocalMaxKernel<<<gdim,bdim,0,self->matutil->stream>>>(pSrc,pDst,oSize,radius);

  return retval;
}
/*
ImageIndexPair maxIdxImageImpl(ImageUtil* self, Image* im, int radius)
{
  Image* dst = self->newEmptyImage(self,im->shape.width,im->shape.height);
  int wregions = im->shape.width/radius;
  int hregions = im->shape.height/radius;
  Npp32f* pSrc = im->pixels->devicePtr;
  Npp32f* pDst = dst->pixels->devicePtr;
  int* pIdx;
  cudaMalloc(&pIdx,sizeof(int)*wregions*hregions);
  NppiSize oSize = im->shape;

  int bdimX = fmin(32,wregions);
  int bdimY = fmin(32,hregions);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(wregions/bdimX,hregions/bdimY);
  //LocalMaxIdxKernel<<<gdim,bdim>>>(pSrc,pDst,pIdx,oSize,radius);

  ImageIndexPair retval = {dst,pIdx};

  return retval;
}*/
//##############
//END STATISTICS
//##############



ImageUtil* GetCUDAImageUtil(MatrixUtil* matutil)
{
  ImageUtil* iu = (ImageUtil*)malloc(sizeof(ImageUtil));

  iu->matutil = matutil;
  iu->newEmptyImage = newEmptyImageImpl;
  iu->newImage = newImageImpl;
  iu->resample = resampleImageImpl;
  iu->add = addImageImpl;
  iu->subtract = subtractImageImpl;
  iu->multiplyC = multiplyCImageImpl;
  iu->max = maxImageImpl;
  //iu->maxIdx = maxIdxImageImpl;
  iu->gradientMagnitude = gradientMagnitudeImageImpl;
  iu->gradientAngle = gradientAngleImageImpl;
  iu->gradients = gradientsImageImpl;
  iu->loadImageFromFile = loadImageFromFileImpl;
  iu->saveImageToFile = saveImageToFileImpl;
  iu->convolve = convolveImageImpl;

  return iu;
}
