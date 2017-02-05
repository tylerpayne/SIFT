#include <utils/MatrixUtil.h>
#include <utils/ImageUtil.h>

#ifndef _FILTERS_
#define _FILTERS_

typedef struct Filters Filters;

typedef Image* (*makeGaussianKernelFunc)(Filters* self, int, float);

struct Filters
{
  ImageUtil* imutil;
  makeGaussianKernelFunc makeGaussianKernel;
};

DLLEXPORT Filters* NewFilters(ImageUtil* imutil);
#endif
