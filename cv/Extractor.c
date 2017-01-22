#include "Extractor.h"

//(Keypoint*)Array.ptr
Array* findCornerKeypointsImpl(Extractor* self, Image* im, int gaussWidth, float gauss1Sigma, float gauss2Sigma, int localMaxWindow, float* contrastTreshold)
{
  if (contrastTreshold == NULL)
  {
    float zzz = 0.2;
    contrastTreshold = &zzz;
  }
  Image* contrast = self->imutil->localContrast(self->imutil,im,localMaxWindow);
  contrast->syncHostFromDevice(contrast);
  Image* gauss1 = self->filters->makeGaussianKernel(self->filters,gaussWidth,gauss1Sigma);
  Image* gauss2 = self->filters->makeGaussianKernel(self->filters,gaussWidth,gauss2Sigma);
  Image* DoGKernel = self->imutil->subtract(self->imutil,gauss1,gauss2);
  gauss1->free(gauss1);
  gauss2->free(gauss2);
  Image* DoGImage = self->imutil->convolve(self->imutil,im,DoGKernel);
  DoGKernel->free(DoGKernel);
  ImageIndexPair* corners = self->imutil->maxIdx(self->imutil,DoGImage,localMaxWindow);
  self->imutil->subPixelAlignImageIndexPair(self->imutil,corners);
  Image* angle = self->imutil->gradientAngle(self->imutil,im);
  Matrix** features = self->imutil->makeFeatureDescriptorsForImageIndexPair(self->imutil,corners,angle,8);
  angle->free(angle);
  Keypoint** keypoints = (Keypoint**)malloc(sizeof(Keypoint*)*corners->count);
  int TOTALCOUNT = 0;
  for (int i = 0; i < corners->count; i++)
  {
    int* idx = C2IDX(corners->index[i],im->shape[0]);
    float X = corners->subPixelX[i];
    float Y = corners->subPixelY[i];
    printf("\nindex: %i subpixel: (%f,%f)\n",corners->index[i],X,Y);
    if (idx[0] <= 0 || idx[1] <= 0)
    {
      continue;
    }
    if (idx[0] >= contrast->shape[1] || idx[1] >= contrast->shape[0])
    {
      continue;
    }
    float con = contrast->pixels->getElement(contrast->pixels,idx[0],idx[1]);
    printf("Contrast: %f",con);
    if (con < contrastTreshold[0] || con != con)
    {
      continue;
    }
    Keypoint* kp = NewKeypoint(X,Y,im);
    kp->set(kp,"nWindowWidth",(void*)&localMaxWindow);
    kp->set(kp,"feature",(void*)(features[i]));
    keypoints[TOTALCOUNT] = kp;
    TOTALCOUNT++;
    free(idx);
  }
  contrast->free(contrast);
  corners->image->free(corners->image);
  free(corners->index);
  free(corners);
  printf("\n Out of the loop, count=%i\n",TOTALCOUNT);
  Keypoint** retkeys = (Keypoint**)malloc(sizeof(Keypoint*)*TOTALCOUNT);
  memcpy(retkeys,keypoints,sizeof(Keypoint*)*TOTALCOUNT);

  Array* retval = (Array*)malloc(sizeof(Array));
  retval->ptr = (void*)retkeys;
  retval->count = TOTALCOUNT;
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
  size_t size = sizeof(float)*featureDim;
  for (int i = 0; i < keypoints->count; i++)
  {
    Keypoint* kp = ((Keypoint**)keypoints->ptr)[i];
    Matrix* feat = (Matrix*)(kp->get(kp,"feature"));
    featureMatrix->setRegion(featureMatrix,i,0,1,featureDim,feat->getRegion(feat,0,0,feat->shape[0],feat->shape[1]));
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
