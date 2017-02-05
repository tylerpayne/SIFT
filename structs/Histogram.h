#include <structs/stdlib.h>
#include <structs/List.h>

#ifndef _HISTOGRAM_
#define _HISTOGRAM_

typedef struct Histogram Histogram;

typedef void (*addFunc)(Histogram*,float);
typedef int (*maxBinFunc)(Histogram*);
typedef void (*tossOutliersFunc)(Histogram*,int, float);

struct Histogram
{
  float binRange;
  int nbins;
  float* binTotals;
  List* bins;
  addFunc add;
  addFunc addTrilinearInterpolate;
  maxBinFunc maxBin;
  maxBinFunc minBin;
  tossOutliersFunc tossOutliers;
};

DLLEXPORT Histogram* NewHistogram(float range,int nbins);

#endif
