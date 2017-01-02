#include "Extractor.h"
#include "Filters.c"
#include "structs/Keypoint.c"

//(Keypoint*)Array.ptr
Array findCornerKeypointsImpl(Extractor* self, Image* im, int gaussWidth, float gauss1Sigma, float gauss2Sigma, int localMaxWindow)
{
  Image* gauss1 = self->filters->makeGaussianKernel(self->filters,gaussWidth,gauss1Sigma);
  Image* gauss2 = self->filters->makeGaussianKernel(self->filters,gaussWidth,gauss2Sigma);
  Image* DoGKernel = self->imutil->subtract(self->imutil,gauss1,gauss2);
  Image* DoGImage = self->imutil->convolve(self->imutil,im,DoGKernel);
  ImageIndexPair corners = self->imutil->maxIdx(self->imutil,DoGImage,localMaxWindow);
  Keypoint* keypoints = (Keypoint*)malloc(sizeof(Keypoint)*corners.n);
  for (int i = 0; i < corners.n; i++)
  {
    int* idx = C2IDX(corners.index[i],im.shape.width);
    Keypoint kp = NewKeypoint((float)idx[1],(float)idx[0],im);
    kp->set(kp,"nWindow",(void*)&localMaxWindow);
    keypoints[i] = kp;
  }
  Array retval = Array{(void*)keypoints,corners.n};
  return retval;
}
//Assign Keypoint Orientations

//Make SIFT Descriptors

Extractor* NewExtractor(ImageUtil* imutil)
{
  Extractor* self = (Extractor*)malloc(sizeof(Extractor));
  self->imutil = imutil;
  Filters* filters = NewFilters(imutil);
  self->filters = filters;
  self->findCornerKeypoints = findCornerKeypointsImpl;

  return self;
}
