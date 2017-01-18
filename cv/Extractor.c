#include "Extractor.h"

//(Keypoint*)Array.ptr
Array* findCornerKeypointsImpl(Extractor* self, Image* im, int gaussWidth, float gauss1Sigma, float gauss2Sigma, int localMaxWindow, float* contrastTreshold)
{
  if (contrastTreshold == NULL)
  {
    float zzz = 0.2;
    contrastTreshold = &zzz;
  }
  Image* gauss1 = self->filters->makeGaussianKernel(self->filters,gaussWidth,gauss1Sigma);
  Image* gauss2 = self->filters->makeGaussianKernel(self->filters,gaussWidth,gauss2Sigma);
  Image* DoGKernel = self->imutil->subtract(self->imutil,gauss1,gauss2);
  Image* DoGImage = self->imutil->convolve(self->imutil,im,DoGKernel);
  ImageIndexPair* corners = self->imutil->maxIdx(self->imutil,DoGImage,localMaxWindow);
  self->imutil->subPixelAlignImageIndexPair(self,corners);
  Image* contrast = self->imutil->localContrast(self->imutil,im,localMaxWindow);
  Keypoint** keypoints = (Keypoint**)malloc(sizeof(Keypoint*)*corners->count);
  int TOTALCOUNT = 0;
  for (int i = 0; i < corners->count; i++)
  {
    int* idx = C2IDX(corners->index[i],im->shape[0]);
    float X = data->subPixelX[i];
    float Y = data->subPixelY[i];
    if (idx[0] <= 0 || idx[1] <= 0)
    {
      continue;
    }
    if (idx[0] >= contrast->shape[1] || idx[1] >= contrast->shape[0])
    {
      continue;
    }
    float con = contrast->pixels->getElement(contrast->pixels,idx[0],idx[1]);
    if (con < contrastTreshold[0] || con != con)
    {
      continue;
    }
    Keypoint* kp = NewKeypoint(X,Y,im);
    kp->set(kp,"nWindow",(void*)&localMaxWindow);
    keypoints[TOTALCOUNT] = kp;
    TOTALCOUNT++;
    free(idx);
  }
  Keypoint** retkeys = (Keypoint**)malloc(sizeof(Keypoint*)*TOTALCOUNT);
  memcpy(retkeys,keypoints,sizeof(Keypoint*)*TOTALCOUNT);
  Array* retval = (Array*)malloc(sizeof(Array));
  retval->ptr = (void*)retkeys;
  retval->count = TOTALCOUNT;

  gauss1->free(gauss1);
  gauss2->free(gauss2);
  DoGKernel->free(DoGKernel);
  DoGImage->free(DoGImage);
  contrast->free(contrast);
  free(keypoints);
  return retval;
}

Matrix* makeFeatureMatrixFromKeypointDescriptorsImpl(Extractor* self, Array* keypoints)
{
  int featureDim;
  Keypoint* kp = ((Keypoint**)keypoints->ptr)[0];
  Matrix* feat = (Matrix*)(kp->get(kp,"feature"));
  featureDim = feat->shape[0]*feat->shape[1];
  Matrix* featureMatrix = self->matutil->newEmptyMatrix(keypoints->count,featureDim);
  copyHostToDeviceCudaMatrix(self->matutil,featureMatrix);
  size_t size = sizeof(float)*featureDim;
  for (int i = 0; i < keypoints->count; i++)
  {
    Keypoint* kp = ((Keypoint**)keypoints->ptr)[i];
    Matrix* feat = (Matrix*)(kp->get(kp,"feature"));
    cudaMemcpy((featureMatrix->devicePtr)+(i*featureDim),feat->devicePtr,size,cudaMemcpyDeviceToDevice);
  }
  return featureMatrix;
}

Extractor* NewExtractor(ImageUtil* imutil)
{
  Extractor* self = (Extractor*)malloc(sizeof(Extractor));
  self->imutil = imutil;
  self->matutil = imutil->matutil;
  Filters* filters = NewFilters(imutil);
  self->filters = filters;
  self->findCornerKeypoints = findCornerKeypointsImpl;
  self->makeFeatureMatrixFromKeypointDescriptors = makeFeatureMatrixFromKeypointDescriptorsImpl;
//  self->makeFeatureDescriptor = makeFeatureDescriptorImpl;
  return self;
}
