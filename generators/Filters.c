#include <generators/Filters.h>

Image* makeGaussianKernelImpl(Filters* self, int width, float std)
{
  float* data = (float*)malloc(sizeof(float)*width*width);
  int radius = width/2;
  float variance = std*std;
  //float coeff = 1.0/(2*M_PI);
  float sum = 0;
  Shape shape = {width,width};
  for (int j = 0; j < width; j++)
  {
    for (int i = 0; i < width; i++)
    {
      float x = j-radius;
      float y = i-radius;
      float power = -1.0*(((x*x)+(y*y))/(2*variance));
      float val = exp(power);
      Point2 point = {j,i};
      data[IDX2C(point,shape)] = val;
      sum += val;
    }
  }
  for (int r = 0; r < width*width; r++)
  {
    data[r] = data[r]/sum;
  }

  return self->imutil->newImage(self->imutil,data,shape);
}

DLLEXPORT Filters* NewFilters(ImageUtil* imutil)
{
  Filters* self = (Filters*)malloc(sizeof(Filters));
  self->imutil = imutil;
  self->makeGaussianKernel = makeGaussianKernelImpl;
  return self;
}
/*
Image* MakeSobelKernels(ImageUtil* imutil)
{
  Image* sobel = (Image*)malloc(sizeof(Image)*2);

  float* x = (float*)malloc(sizeof(float)*9);
  float* y = (float*)malloc(sizeof(float)*9);

  x[0] = -1.0;
  x[1] = 0;
  x[2] = 1.0;
  x[3] = -2.0;
  x[4] = 0;
  x[5] = 2.0;
  x[6] = -1.0;
  x[7]  = 0.0;
  x[8] = 1.0;

  y[0] = -1.0;
  y[1] = -2.0;
  y[2] = -1.0;
  y[3] = 0.0;
  y[4] = 0;
  y[5] = 0.0;
  y[6] = 1.0;
  y[7] = 2.0;
  y[8] = 1.0;

  Matrix* matx = imutil->matrixUtil->newMatrix(x,3,3);
  Matrix* maty = imutil->matrixUtil->newMatrix(y,3,3);

  Image* sobelx = imutil->newImageFromMatrix(imutil,matx);
  Image* sobely = imutil->newImageFromMatrix(imutil,maty);

  sobel[0] = *sobelx;
  sobel[1] = *sobely;

  return sobel;
}*/
