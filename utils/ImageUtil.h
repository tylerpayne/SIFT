#include <structs/Image.h>

#ifndef _IMAGEUTIL_
#define _IMAGEUTIL_

typedef struct ImageUtil ImageUtil;

typedef Image* (*newImageFromHostFloatFunc)(ImageUtil*,float*,Shape);
typedef Image* (*newImageFromMatrixFunc)(ImageUtil*, Matrix*);
typedef Image* (*newEmptyImageFunc)(ImageUtil*, Shape);
typedef Image* (*downsampleImageFunc)(ImageUtil*,Image*,Shape);


typedef void (*imFunc)(ImageUtil*,Image*);
typedef void (*imimFunc)(ImageUtil*,Image*,Image*);
typedef void (*imimimFunc)(ImageUtil*,Image*,Image*,Image*);
typedef void (*imiimFunc)(ImageUtil*,Image*,int,Image*);
typedef void (*imfimFunc)(ImageUtil*,Image*,float,Image*);
typedef void (*imimimimimFunc)(ImageUtil*,Image*,Image*,Image*,Image*,Image*);

typedef void (*imshapeimFunc)(ImageUtil*,Image*,Shape,Image*);

typedef void (*im)

typedef ImageIndexPair* (*im1iim2p2f)(ImageUtil*,Image*,int);

typedef ImageGradientVectorPair* (*rIMGVimFunc)(ImageUtil*,Image*);
typedef ImageIndexPair* (*IMIDXfloatFunc)(ImageUtil*,Image*,float);
typedef void (*subPixelAlignImageIndexPairFunc)(ImageUtil*,ImageIndexPair*);
typedef Matrix* (*makeFeatureDescriptorFunc)(ImageUtil*,Image*,int*,int);
typedef Matrix** (*makeFeatureDescriptorsForImageIndexPairFunc)(ImageUtil*,ImageIndexPair*,Image*,int);
typedef void (*unorientFeatureMatrixFunc)(ImageUtil*, Matrix*, int);
typedef Matrix* (*generalizeFeatureMatrixFunc)(ImageUtil*, Matrix*, int);
typedef void (*eliminatePointsFunc)(ImageUtil*,ImageIndexPair*,float*);


struct ImageUtil
{
  MatrixUtil* matutil;
  newEmptyImageFunc newEmptyImage;
  newImageFromHostFloatFunc newImage;
  newImageFromMatrixFunc newImageFromMatrix;

  imimimFunc add, subtract, multiply, divide, convolve;
  imfimFunc addf, subtractf, multiplyf, dividef, threshold;
  imiim max, argmax, localContrast;
  imimimimimFunc gradients;

  imshapeimFunc resample;

  imintFunc localContrast;
  IMIDXfloatFunc thresholdIdx;

  subPixelAlignImageIndexPairFunc subPixelAlignImageIndexPair;
  makeFeatureDescriptorFunc makeFeatureDescriptor;
  makeFeatureDescriptorsForImageIndexPairFunc makeFeatureDescriptorsForImageIndexPair;
  unorientFeatureMatrixFunc unorientFeatureMatrix;
  generalizeFeatureMatrixFunc generalizeFeatureMatrix;
  eliminatePointsFunc eliminatePointsBelowThreshold;
  eliminatePointsFunc eliminateEdgePoints;
};

DLLEXPORT ImageUtil* GetImageUtil(MatrixUtil* matutil);

#endif
