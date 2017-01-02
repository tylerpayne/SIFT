typedef struct Filters Filters;

typedef Image* (*makeGaussianKernelFunc)(Filters* self, int, float);

struct Filters
{
  ImageUtil* imutil;
  makeGaussianKernelFunc makeGaussianKernel;
};

Filters* NewFilters(ImageUtil* imutil);
