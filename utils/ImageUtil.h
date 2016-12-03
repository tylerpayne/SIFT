#include <stdlib.h>
//#include "MatrixUtil.h"
#include "Image.h"
#include "lodepng/lodepng.c"

typedef struct ImageUtil ImageUtil;

typedef Image* (*newImageFromMatrixFunc)(ImageUtil*,Matrix*);
typedef Image* (*newEmptyImageFunc)(ImageUtil*, int, int );
typedef Image* (*loadImageFromFileFunc)(ImageUtil*, char*);
typedef void (*saveImageToFileFunc)(ImageUtil*,Image*, char*);
typedef ScaleSpaceImage* (*buildPyramidFunc)(ImageUtil*,Image*,int);
typedef ScaleSpaceImage* (*newScaleSpaceImageFunc)(ImageUtil*,int,int);
typedef ScaleSpaceImage* (*ssssFunc)(ImageUtil*,ScaleSpaceImage*);
typedef Image* (*downsampleImageFunc)(ImageUtil*,Image*,int,int);
typedef Image* (*imimFunc)(ImageUtil*,Image*,Image*);
typedef Image* (*generateGaussFunc)(ImageUtil*,int,float);
typedef void (*ssISyncFunc)(ImageUtil*, ScaleSpaceImage*);

struct ImageUtil
{
  MatrixUtil* matrixUtil;
  newImageFromMatrixFunc newImageFromMatrix;
  newEmptyImageFunc newEmptyImage;
  downsampleImageFunc downsample;
  loadImageFromFileFunc loadImageFromFile;
  saveImageToFileFunc saveImageToFile;
  buildPyramidFunc buildPyrmaid;
  imimFunc convolve;
  newScaleSpaceImageFunc newEmptyScaleSpaceImage;
  ssISyncFunc syncScaleSpaceImage;
};

ImageUtil* GetImageUtil(MatrixUtil* matrixUtil);
