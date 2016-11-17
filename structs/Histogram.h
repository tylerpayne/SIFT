#include <stdlib.h>
#include "LinkedList.h"

typedef struct Histogram Histogram;

typedef void (*addFunc)(Histogram*,float);
typedef int (*maxBinFunc)(Histogram*);
typedef void (*tossOutliersFunc)(Histogram*,int, float);

struct Histogram
{
  float binRange;
  int nbins;
  float* binTotals;
  LinkedList* bins;
  addFunc add;
  addFunc addTrilinearInterpolate;
  maxBinFunc maxBin;
  maxBinFunc minBin;
  tossOutliersFunc tossOutliers;
};

Histogram* NewHistogram(float range,int nbins);
