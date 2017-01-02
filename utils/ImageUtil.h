#include <stdlib.h>
#include <math.h>
#include <nppcore.h>

#include "Image.h"
#include "lodepng/lodepng.c"
#include "ImageKernels.cu"

typedef struct ImageIndexPair ImageIndexPair;

struct ImageIndexPair
{
  Image* image;
  int* index;
  int n;
};

typedef struct ImageGradientVectorPair ImageGradientVectorPair;

struct ImageGradientVectorPair
{
  Image* magnitude;
  Image* angle;
};

typedef struct ImageUtil ImageUtil;

typedef Image* (*newImageFromHostFloatFunc)(ImageUtil*,float*,int,int);
typedef Image* (*newEmptyImageFunc)(ImageUtil*, int, int );
typedef Image* (*loadImageFromFileFunc)(ImageUtil*, char*);
typedef void (*saveImageToFileFunc)(ImageUtil*,Image*, char*);
typedef Image* (*downsampleImageFunc)(ImageUtil*,Image*,int,int);
typedef Image* (*imimFunc)(ImageUtil*,Image*,Image*);
typedef Image* (*imintFunc)(ImageUtil*,Image*,int);
typedef ImageIndexPair (*rIMIDXimintFunc)(ImageUtil*,Image*,int);
typedef Image* (*imFunc)(ImageUtil*,Image*);
typedef ImageGradientVectorPair (*rIMGVimFunc)(ImageUtil*,Image*);
typedef Image* (*imfloatFunc)(ImageUtil*,Image*,float);
typedef Image* (*generateGaussFunc)(ImageUtil*,int,float);

struct ImageUtil
{
  MatrixUtil* matutil;
  newEmptyImageFunc newEmptyImage;
  newImageFromHostFloatFunc newImage;
  downsampleImageFunc resample;
  loadImageFromFileFunc loadImageFromFile;
  saveImageToFileFunc saveImageToFile;
  imimFunc convolve;
  imimFunc add;
  imimFunc subtract;
  imfloatFunc multiplyC;
  imintFunc max;
  rIMIDXimintFunc maxIdx;
  imFunc gradientMagnitude;
  imFunc gradientAngle;
  rIMGVimFunc gradients;
};

ImageUtil* GetImageUtil(MatrixUtil* matutil);
ImageUtil* GetCUDAImageUtil(MatrixUtil* matutil);
