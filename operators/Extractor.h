#include <utils/MatrixUtil.h>
#include <utils/ImageUtil.h>
#include <generators/Filters.h>
#include <structs/Keypoint.h>

#ifndef _EXTRACTOR_
#define _EXTRACTOR_

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

DLLEXPORT Extractor* NewExtractor(ImageUtil* imutil);
#endif
