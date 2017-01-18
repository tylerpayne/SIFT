#include "cv/Filters.c"

typedef struct Extractor Extractor;

typedef Array* (*findCornerKeypointsFunc)(Extractor*, Image*, int, float, float, int,float*);
typedef Matrix* (*makeFeatureMatrixFromKeypointDescriptorsFunc)(Extractor*,Array*);

struct Extractor
{
  ImageUtil* imutil;
  MatrixUtil* matutil;
  Filters* filters;
  findCornerKeypointsFunc findCornerKeypoints;
  makeFeatureMatrixFromKeypointDescriptorsFunc makeFeatureMatrixFromKeypointDescriptors;

};

Extractor* NewExtractor(ImageUtil* imutil);
