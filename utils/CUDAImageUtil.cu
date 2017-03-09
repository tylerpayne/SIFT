#include <utils/MatrixUtil.h>
#include <utils/ImageUtil.h>
#include <nppi.h>
#include "kernels/ImageKernels.cu"

#ifdef __cplusplus
  extern "C" {
#endif

void cudaSafeCall(cudaError_t stat)
{
  if (stat != cudaSuccess)
  {
    printf("CUDA ERR\n%s\n",cudaGetErrorString(stat));
  }
}

void nppSafeCall(NppStatus status)
{
  if (status != NPP_SUCCESS)
  {
    printf("\n##########\nNPP ERROR!\nError code: %d\n##########\n",status);
  }
}

//#############
//INIT Methods
//############

void freeImageImpl(Image* self)
{
  self->pixels->free(self->pixels);
  free(self->pixbuf);
  free(self);
}

Image* newEmptyImageImpl(ImageUtil* self, Shape shape)
{
  Image* im = (Image*)malloc(sizeof(Image));
  im->nChannels = 1;
  im->shape = shape;
  im->pixels=self->matutil->newEmptyMatrix(shape);
  im->free = freeImageImpl;
  return im;
}

Image* newImageImpl(ImageUtil* self, float* data, Shape shape)
{
  Image* im = self->newEmptyImage(self,shape);
  free(im->pixels->hostPtr);
  im->pixels->hostPtr = data;
  return im;
}

Image* newImageFromMatrixImpl(ImageUtil* self, Matrix* m)
{
  Image* im = self->newEmptyImage(self,m->shape);
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
Image* convolveImageImpl(ImageUtil* self, Image* A, Image* B, Image* C)
{
  assert(!A->pixels->isHostSide);
  assert(!B->pixels->isHostSide);
  assert(!C->pixels->isHostSide);

  Npp32f* pSrc = A->pixels->devicePtr;
  int nSrcStep = A->shape.height*sizeof(float);
  NppiSize oSrcSize = {A->shape.height,A->shape.width};
  NppiPoint oSrcOffset = {0,0};

  Npp32f* pDst = C->pixels->devicePtr;
  int nDstStep = A->shape.height*sizeof(float);
  NppiSize oSizeROI = {A->shape.height,A->shape.width};

  Npp32f* pKernel = B->pixels->devicePtr;
  NppiSize oKernelSize = {B->shape.height,B->shape.width};
  NppiPoint oAnchor = {oKernelSize.width/2,oKernelSize.height/2};

  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

  nppSafeCall(nppiFilterBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,
                                      pDst,nDstStep,oSizeROI,
                                      pKernel,oKernelSize,oAnchor,
                                      eBorderType));
}

void gradientsImageImpl(ImageUtil* self, Image* src, Image* dX, Image* dY, Image* mag, Image* angle)
{
  assert(!src->pixels->isHostSide);

  Npp32f* pSrc = src->pixels->devicePtr;
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
    assert(!dX->pixels->isHostSide);
    pDstX = dX->pixels->devicePtr;
    nDstXStep = nSrcStep;
  }

  if (dY != NULL)
  {
    assert(!dY->pixels->isHostSide);
    pDstY = dY->pixels->devicePtr;
    nDstYStep = nSrcStep;
  }

  if (mag != NULL)
  {
    assert(!mag->pixels->isHostSide);
    pDstMag = mag->pixels->devicePtr;
    nDstMagStep = nSrcStep;
  }

  if (angle != NULL)
  {
    assert(!angle->pixels->isHostSide);
    pDstAngle = angle->pixels->devicePtr;
    nDstAngleStep = nSrcStep;
  }

  NppiSize oSizeROI = {src->shape.height,src->shape.width};
  NppiNorm eNorm = nppiNormL2;
  NppiMaskSize eMaskSize = NPP_MASK_SIZE_3_X_3;
  NppiBorderType eBorderType = NPP_BORDER_REPLICATE;

  nppSafeCall(nppiGradientVectorSobelBorder_32f_C1R(pSrc,nSrcStep,oSrcSize,oSrcOffset,
                                                    pDstX,nDstXStep,
                                                    pDstY,nDstYStep,
                                                    pDstMag,nDstMagStep,
                                                    pDstAngle,nDstAngleStep,
                                                    oSizeROI,eMaskSize,eNorm,eBorderType));
}
//############
//END FILTERS
//###########

//#########
//GEOMETRY
//########
void resampleImageImpl(ImageUtil* self, Image* A, Shape shape, Image* C)
{
  assert(!A->pixels->isHostSide);
  assert(!C->pixels->isHostSide);

  Npp32f* pSrc = A->pixels->devicePtr;
  int nSrcStep = A->shape.height*sizeof(float);
  NppiSize oSrcSize = {A->shape.height,A->shape.width};
  NppiRect oSrcROI = {0,0,A->shape.height,A->shape.width};

  Npp32f* pDst = C->pixels->devicePtr;
  int nDstStep = C->shape.height*sizeof(float);
  NppiRect oDstROI = {0,0,C->shape.height,C->shape.width};

  double nXFactor = ((float)shape.width)/((float)A->shape.height);
  double nYFactor = ((float)shape.height)/((float)A->shape.width);
  double nXShift = 0;
  double nYShift = 0;
  NppiInterpolationMode eInterpolation = NPPI_INTER_CUBIC;

  nppSafeCall(nppiResizeSqrPixel_32f_C1R(pSrc,oSrcSize,nSrcStep,oSrcROI,
                                        pDst,nDstStep,oDstROI,
                                        nXFactor,nYFactor,
                                        nXShift,nYShift,
                                        eInterpolation));
}
//#############
//END GEOMETRY
//#############

//##########
//ARITHMETIC
//##########
void addImageImpl(ImageUtil* self, Image* A, Image* B, Image* C)
{
  self->matutil->add(A->pixels,B->pixels,C->pixels);
}

void addfImageImpl(ImageUtil* self, Image* A, float b, Image* C)
{
  self->matutil->addf(A->pixels,b,C->pixels);
}

void subtractImageImpl(ImageUtil* self, Image* A, Image* B, Image* C)
{
  self->matutil->subtract(A->pixels,B->pixels,C->pixels);
}

void subtractfImageImpl(ImageUtil* self, Image* A, float b, Image* C)
{
  self->matutil->subtractf(A->pixels,b,C->pixels);
}

void multiplyImageImpl(ImageUtil* self, Image* A, Image* B, Image* C)
{
  self->matutil->multiply(A->pixels,B->pixels,C->pixels);
}

void multiplyfImageImpl(ImageUtil* self, Image* A, float b, Image* C)
{
  self->matutil->multiplyf(A->pixels,b,C->pixels);
}

void divideImageImpl(ImageUtil* self, Image* A, Image* B, Image* C)
{
  self->matutil->divide(A->pixels,B->pixels,C->pixels);
}

void dividefImageImpl(ImageUtil* self, Image* A, float b, Image* C)
{
  self->matutil->dividef(A->pixels,b,C->pixels);
}
//###############
//END ARITHMETIC
//##############

//##########
//STATISTICS
//##########
void maxImageImpl(ImageUtil* self, Image* A, int r, Image* C)
{
  assert(!A->pixels->isHostSide);
  assert(!C->pixels->isHostSide);

  int wregions = A->shape.height/r;
  int hregions = A->shape.width/r;
  float* pSrc = A->pixels->devicePtr;
  float* pDst = C->pixels->devicePtr;

  int bdimX = fmin(sqrt(THREADS_PER_BLOCK),wregions);
  int bdimY = fmin(sqrt(THREADS_PER_BLOCK),hregions);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(wregions/bdimX + 1,hregions/bdimY + 1);
  LocalMaxKernel<<<gdim,bdim,0,_stream>>>(pSrc,pDst,A->shape,r);
}

void argmaxImageImpl(ImageUtil* self, Image* A, int r, Point2** pMax)
{
  assert(!A->pixels->isHostSide);

  int wregions = A->shape.height/r;
  int hregions = A->shape.width/r;
  float* pSrc = A->pixels->devicePtr;

  cudaSafeCall(cudaMalloc(pMax,sizeof(Point2)*wregions*hregions));

  int bdimX = fmin(sqrt(THREADS_PER_BLOCK),wregions);
  int bdimY = fmin(sqrt(THREADS_PER_BLOCK),hregions);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(wregions/bdimX + 1,hregions/bdimY + 1);
  LocalArgMaxKernel<<<gdim,bdim,0,_stream>>>(pSrc,(*pMax),A->shape,r);
}

void localContrastImageImpl(ImageUtil* self, Image* A, int radius, Image* C)
{
  assert(!A->pixels->isHostSide);
  assert(!C->pixels->isHostSide);

  int wregions = A->shape.height/r;
  int hregions = A->shape.width/r;
  float* pSrc = A->pixels->devicePtr;
  float* pDst = C->pixels->devicePtr;

  int bdimX = fmin(sqrt(THREADS_PER_BLOCK),wregions);
  int bdimY = fmin(sqrt(THREADS_PER_BLOCK),hregions);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(wregions/bdimX + 1,hregions/bdimY + 1);
  LocalContrastKernel<<<gdim,bdim,0,_stream>>>(pSrc,pDst,A->shape,r);
}
//##############
//END STATISTICS
//##############

//######################
// BEGIN COMPUTERVISION
//######################
void subPixelRefineImageImpl(ImageUtil* self, Image* A, Point2* pMax, Point2f** pSubPixel)
{
  assert(!A->pixels->isHostSide);

  Image* Ix = self->gradientX(self,data->image);
  Image* Iy = self->gradientY(self,data->image);
  Image* Ixx = self->gradientX(self,Ix);
  Image* Ixy = self->gradientY(self,Ix);
  Image* Iyy = self->gradientY(self,Iy);

  Shape oSize = {Ix->shape.height,Ix->shape.width};

  size_t size = sizeof(Point2f)*(data->count);
  cudaSafeCall(cudaMalloc(pSubPixel,size));

  int bdimX = min(THREADS_PER_BLOCK,data->count);
  dim3 bdim(bdimX);
  dim3 gdim((data->count/bdimX) + 1);
  SubPixelAlignKernel<<<gdim,bdim,0,_stream>>>(Ix->pixels->devicePtr,Iy->pixels->devicePtr,Ixx->pixels->devicePtr,Ixy->pixels->devicePtr,Iyy->pixels->devicePtr,pIdx,*pSubPixel,oSize,data->count);

  Ix->free(Ix);
  Iy->free(Iy);
  Ixx->free(Ixx);
  Ixy->free(Ixy);
  Iyy->free(Iyy);
}

void eliminatePointsBelowThresholdImpl(ImageUtil* self, ImageIndexPair* keypoints, float* threshold)
{
  printf("ELIM BELOW THRESHOLD\n");
  float thresh;
  if (threshold == NULL)
  {
    thresh = 0.1;
  } else {
    thresh = *threshold;
  }
  printf("1\n");
  int* d_keepCount;
  int* d_keepIndex;
  Point2f* d_keepSubPixel;

  cudaSafeCall(cudaMalloc(&d_keepIndex,sizeof(int)*keypoints->count));
  cudaSafeCall(cudaMalloc(&d_keepSubPixel,sizeof(Point2f)*keypoints->count));
  cudaSafeCall(cudaMalloc(&d_keepCount,sizeof(int)));
  cudaSafeCall(cudaMemset(d_keepCount,0,sizeof(int)));
  printf("2\n");

  NppiSize oSize = {keypoints->image->shape.height,keypoints->image->shape.width};

  if (keypoints->image->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(keypoints->image->pixels);
  }

  int blockDimX = fmin(1024,keypoints->count);
  dim3 bdim(blockDimX);
  dim3 gdim(keypoints->count/blockDimX + 1);
  printf("bdim: (%i,%i) gdim: (%i,%i) N: %i\n",bdim.x,bdim.y,gdim.x,gdim.y,keypoints->count);
  printf("imshape: (%i,%i)\n",keypoints->image->shape.height,keypoints->image->shape.width);
  printf("isNull: %i\n",keypoints->index==NULL);
  EliminatePointsBelowThresholdKernel<<<gdim,bdim,sizeof(int)*keypoints->count+1>>>(keypoints->image->pixels->devicePtr,oSize,keypoints->subPixel,keypoints->index,keypoints->count,d_keepSubPixel,d_keepIndex,d_keepCount,thresh);
  cudaDeviceSynchronize();
  cudaSafeCall(cudaGetLastError());
  printf("3\n");
  int* h_keepCount = (int*)malloc(sizeof(int));
  cudaSafeCall(cudaMemcpy(h_keepCount,d_keepCount,sizeof(int),cudaMemcpyDeviceToHost));
  int keepCount = *h_keepCount;
  printf("h_keepCount = %i\n",keepCount);
  cudaSafeCall(cudaFree(&d_keepIndex[keepCount]));
  cudaSafeCall(cudaFree(&d_keepSubPixel[keepCount]));
  printf("4\n");

  keypoints->count = keepCount;
  keypoints->index = d_keepIndex;
  keypoints->subPixel = d_keepSubPixel;
}

void eliminateEdgePointsImpl(ImageUtil* self, ImageIndexPair* keypoints, float* threshold)
{
  float thresh;
  if (threshold == NULL)
  {
    thresh = 0.1;
  } else {
    thresh = *threshold;
  }

  Image* Ix = self->gradientX(self,keypoints->image);
  Image* Iy = self->gradientY(self,keypoints->image);

  int* d_keepCount;
  int* d_keepIndex;
  Point2f* d_keepSubPixel;
  cudaSafeCall(cudaMalloc(&d_keepCount,sizeof(int)));
  cudaSafeCall(cudaMalloc(&d_keepIndex,sizeof(int)*keypoints->count));
  cudaSafeCall(cudaMalloc(&d_keepSubPixel,sizeof(Point2f)*keypoints->count));

  NppiSize oSize = {keypoints->image->shape.height,keypoints->image->shape.width};

  if (keypoints->image->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(keypoints->image->pixels);
  }

  if (Ix->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(Ix->pixels);
  }

  if (Iy->pixels->isHostSide)
  {
    copyHostToDeviceCudaMatrix(Iy->pixels);
  }

  int blockDimX = fmin(1024,keypoints->count);

  dim3 bdim(blockDimX);
  dim3 gdim(keypoints->count/blockDimX + 1);
  EliminateEdgePointsKernel<<<gdim,bdim>>>(keypoints->image->pixels->devicePtr,oSize,keypoints->subPixel,keypoints->index,keypoints->count,Ix->pixels->devicePtr,Iy->pixels->devicePtr,d_keepSubPixel,d_keepIndex,d_keepCount,thresh);
  Ix->free(Ix);
  Iy->free(Iy);
  int* h_keepCount = (int*)malloc(sizeof(int));
  cudaSafeCall(cudaMemcpy(h_keepCount,d_keepCount,sizeof(int),cudaMemcpyDeviceToHost));
  int keepCount = h_keepCount[0];
  cudaSafeCall(cudaFree(&d_keepIndex[keepCount]));
  cudaSafeCall(cudaFree(&d_keepSubPixel[keepCount]));

  keypoints->count = keepCount;
  keypoints->index = d_keepIndex;
  keypoints->subPixel = d_keepSubPixel;
}


Matrix** makeFeatureDescriptorsForImageIndexPairImpl(ImageUtil* self, ImageIndexPair* keypoints, Image* im, int featureWidth)
{
  float* d_features;
  float* d_subPixelX;
  float* d_subPixelY;
  cudaSafeCall(cudaMalloc(&d_features,sizeof(float)*featureWidth*featureWidth*keypoints->count));
  cudaSafeCall(cudaMalloc(&d_subPixelY,sizeof(float)*keypoints->count));
  cudaSafeCall(cudaMalloc(&d_subPixelX,sizeof(float)*keypoints->count));
  cudaMemcpy(d_subPixelX,keypoints->subPixel,sizeof(Point2f)*keypoints->count,cudaMemcpyHostToDevice);

  NppiSize oSize = {im->shape.height,im->shape.width};

  int bdimX = min(1024,keypoints->count);
  dim3 bdim(bdimX);
  dim3 gdim((keypoints->count/bdimX) + 1);
  MakeFeatureDescriptorKernel<<<gdim, bdim>>>(im->pixels->devicePtr,oSize,d_subPixelX,d_subPixelY,keypoints->count,d_features,featureWidth);
  Matrix** retval = (Matrix**)malloc(sizeof(Matrix*)*keypoints->count);
  for (int i = 0; i < keypoints->count; i++)
  {
    float* h_feature = (float*)malloc(sizeof(float)*featureWidth*featureWidth);
    cudaSafeCall(cudaMemcpy(h_feature,&d_features[i*featureWidth*featureWidth],sizeof(float)*featureWidth*featureWidth,cudaMemcpyDeviceToHost));
    Shape shape = {featureWidth,featureWidth};
    Matrix* m = self->matutil->newMatrix(h_feature,shape);
    retval[i] = m;
  }
  cudaSafeCall(cudaFree(d_subPixelX));
  cudaSafeCall(cudaFree(d_subPixelY));
  cudaSafeCall(cudaFree(d_features));
  return retval;
}

Matrix* generalizeFeatureMatrixImpl(ImageUtil* self, Matrix* features, int nBins)
{
  if (features->isHostSide)
  {
    copyHostToDeviceCudaMatrix(features);
  }
  int nFeatureWidth = (int)sqrt(features->shape.width);
  Shape shape = {nBins*nFeatureWidth,features->shape.height};
  Matrix* genFeatures = self->matutil->newEmptyMatrix(shape);
  copyHostToDeviceCudaMatrix(genFeatures);
  int bdimX = 16;
  int bdimY = fmin(64,features->shape.height);
  dim3 bdim(bdimX,bdimY);
  dim3 gdim(1,features->shape.height/bdimY + 1);
  GeneralizeFeatureKernel<<<gdim,bdim>>>(features->devicePtr,genFeatures->devicePtr,features->shape.height,features->shape.width,16,nBins);
  return genFeatures;
}

void unorientFeatureMatrixImpl(ImageUtil* self, Matrix* features, int nBins)
{
  if (features->isHostSide)
  {
    copyHostToDeviceCudaMatrix(features);
  }
  int bdimX = fmin(1024,features->shape.height);
  dim3 bdim(bdimX);
  dim3 gdim(features->shape.height/bdimX + 1);
  UnorientFeatureKernel<<<gdim,bdim,sizeof(float)*nBins>>>(features->devicePtr,features->shape.height,features->shape.width,nBins);
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
  self->eliminatePointsBelowThreshold = eliminatePointsBelowThresholdImpl;
  self->eliminateEdgePoints = eliminateEdgePointsImpl;

  return self;
}

#ifdef __cplusplus
  }
#endif
