#include "Image.h"

typedef struct ImageIndexPair ImageIndexPair;

struct ImageIndexPair
{
  Image* image;
  int* index;
  int count;
  float* subPixelX;
  float* subPixelY;
};

typedef struct ImageGradientVectorPair ImageGradientVectorPair;

struct ImageGradientVectorPair
{
  Image* magnitude;
  Image* angle;
};

typedef struct ImageUtil ImageUtil;

typedef Image* (*newImageFromHostFloatFunc)(ImageUtil*,float*,int,int);
typedef Image* (*newImageFromMatrixFunc)(ImageUtil*, Matrix*);
typedef Image* (*newEmptyImageFunc)(ImageUtil*, int, int );
typedef Image* (*downsampleImageFunc)(ImageUtil*,Image*,int,int);
typedef Image* (*imimFunc)(ImageUtil*,Image*,Image*);
typedef Image* (*imintFunc)(ImageUtil*,Image*,int);
typedef ImageIndexPair* (*rIMIDXimintFunc)(ImageUtil*,Image*,int);
typedef Image* (*imFunc)(ImageUtil*,Image*);
typedef ImageGradientVectorPair* (*rIMGVimFunc)(ImageUtil*,Image*);
typedef Image* (*imfloatFunc)(ImageUtil*,Image*,float);
typedef ImageIndexPair* (*IMIDXfloatFunc)(ImageUtil*,Image*,float);
typedef void (*subPixelAlignImageIndexPairFunc)(ImageUtil*,ImageIndexPair*);
typedef Matrix* (*makeFeatureDescriptorFunc)(ImageUtil*,Image*,int*,int);
typedef Matrix** (*makeFeatureDescriptorsForImageIndexPairFunc)(ImageUtil*,ImageIndexPair*,Image*,int);
typedef void (*unorientFeatureMatrixFunc)(ImageUtil*, Matrix*, int);
typedef Matrix* (*generalizeFeatureMatrixFunc)(ImageUtil*, Matrix*, int);


struct ImageUtil
{
  MatrixUtil* matutil;
  newEmptyImageFunc newEmptyImage;
  newImageFromHostFloatFunc newImage;
  newImageFromMatrixFunc newImageFromMatrix;
  downsampleImageFunc resample;
  imimFunc convolve;
  imimFunc add;
  imimFunc subtract;
  imimFunc multiply;
  imfloatFunc multiplyC;
  imintFunc max;
  imintFunc localContrast;
  rIMIDXimintFunc maxIdx;
  IMIDXfloatFunc thresholdIdx;
  imFunc gradientX;
  imFunc gradientY;
  imFunc gradientMagnitude;
  imFunc gradientAngle;
  rIMGVimFunc gradients;
  subPixelAlignImageIndexPairFunc subPixelAlignImageIndexPair;
  makeFeatureDescriptorFunc makeFeatureDescriptor;
  makeFeatureDescriptorsForImageIndexPairFunc makeFeatureDescriptorsForImageIndexPair;
  unorientFeatureMatrixFunc unorientFeatureMatrix;
  generalizeFeatureMatrixFunc generalizeFeatureMatrix;
};

DLLEXPORT ImageUtil* GetImageUtil(MatrixUtil* matutil);
