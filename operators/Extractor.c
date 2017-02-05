#include "Extractor.h"

//(Keypoint*)Array.ptr
Array* findCornerKeypointsImpl(Extractor* self, Image* im, int gaussWidth, float gauss1Sigma, float gauss2Sigma, int localMaxWindow, float* contrastTreshold)
{

  if (contrastTreshold == NULL)
  {
    float zzz = 0.01;
    contrastTreshold = &zzz;
  }

  Image* gauss1 = self->filters->makeGaussianKernel(self->filters,gaussWidth,gauss1Sigma);
  Image* gauss2 = self->filters->makeGaussianKernel(self->filters,gaussWidth,gauss2Sigma);
  Image* DoGKernel = self->imutil->subtract(self->imutil,gauss1,gauss2);
  gauss1->free(gauss1);
  gauss2->free(gauss2);
  Image* DoGImage = self->imutil->convolve(self->imutil,im,DoGKernel);
  DoGKernel->free(DoGKernel);
  ImageIndexPair* corners = self->imutil->maxIdx(self->imutil,DoGImage,localMaxWindow);
  printf("Subpixel start\n");
  self->imutil->subPixelAlignImageIndexPair(self->imutil,corners);
  printf("subpixel\n");
  Image* contrast = self->imutil->localContrast(self->imutil,im,localMaxWindow);
  printf("made contrast image\n");
  corners->image = contrast;
  self->imutil->eliminatePointsBelowThreshold(self->imutil,corners,NULL);
  printf("coontrast\n");
  contrast->free(contrast);
  corners->image = im;
  self->imutil->eliminateEdgePoints(self->imutil,corners,NULL);
  printf("edges\n");
  Image* angle = self->imutil->gradientAngle(self->imutil,im);
  Matrix** features = self->imutil->makeFeatureDescriptorsForImageIndexPair(self->imutil,corners,angle,16);
  angle->free(angle);
  Keypoint** keypoints = (Keypoint**)malloc(sizeof(Keypoint*)*corners->count);
  int TOTALCOUNT = 0;
  for (int i = 0; i < corners->count; i++)
  {
    Point2 idx = C2IDX(corners->index[i],im->shape);
    float X = corners->subPixelX[i];
    float Y = corners->subPixelY[i];
    if (idx.x <= 0 || idx.y <= 0)
    {
      continue;
    }
    Keypoint* kp = NewKeypoint(X,Y,im);
    kp->set(kp,"nWindowWidth",(void*)&localMaxWindow);
    kp->set(kp,"feature",(void*)(features[i]));
    keypoints[TOTALCOUNT] = kp;
    TOTALCOUNT++;
  }
  contrast->free(contrast);
  corners->image->free(corners->image);
  free(corners->index);
  free(corners);
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
  featureDim = feat->shape.width*feat->shape.height;
  Shape shape = {keypoints->count,featureDim};
  Matrix* featureMatrix = self->matutil->newEmptyMatrix(shape);
  size_t size = sizeof(float)*featureDim;
  Point2 origin = {0,0};
  Rect rect = {feat->shape,origin};

  for (int i = 0; i < keypoints->count; i++)
  {
    Keypoint* kp = ((Keypoint**)keypoints->ptr)[i];
    Matrix* feat = (Matrix*)(kp->get(kp,"feature"));
    Point2 or = {i,0};
    Shape sh = {1,featureDim};
    Rect rect = {sh,or};
    featureMatrix->setRegion(featureMatrix,rect,feat->getRegion(feat,rect));
  }

  return self->imutil->generalizeFeatureMatrix(self->imutil,featureMatrix,12);
}

DLLEXPORT Extractor* NewExtractor(ImageUtil* imutil)
{
  Extractor* self = (Extractor*)malloc(sizeof(Extractor));
  self->imutil = imutil;
  self->matutil = imutil->matutil;
  Filters* filters = NewFilters(imutil);
  self->filters = filters;
  self->findCornerKeypoints = findCornerKeypointsImpl;
  self->makeFeatureMatrixFromKeypointDescriptors = makeFeatureMatrixFromKeypointDescriptorsImpl;
  //self->makeFeatureDescriptor = makeFeatureDescriptorImpl;
  return self;
}
