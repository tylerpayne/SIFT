#include "ImageUtil.h"
#include <math.h>

Image* newEmptyImageImpl(ImageUtil* self, int width, int height)
{
  Image* im = (Image*)malloc(sizeof(Image));
  im->nChannels = 1;
  int* shape = (int*)malloc(sizeof(int)*2);
  shape[0] = width;
  shape[1] = height;
  im->shape = shape;

  Matrix* mat = self->matrixUtil->newEmptyMatrix(width,height);
  im->pixels=mat;

  return im;
}

Image* newImageFromMatrixImpl(ImageUtil* self, Matrix* data)
{
  Image* im = (Image*)malloc(sizeof(Image));
  im->nChannels = 1;
  int* shape = (int*)malloc(sizeof(int)*2);
  shape[0] = data->shape[0];
  shape[1] = data->shape[1];
  im->shape = shape;
  im->pixels = data;
  return im;
}

Image* loadImageFromFileImpl(ImageUtil* self, char* p)
{
  unsigned error;
  unsigned char* image;
  unsigned width, height;
  const char* path = p;
  error = lodepng_decode32_file(&image, &height, &width, path);
  if (error)
  {
    printf("error %u: %s\n", error, lodepng_error_text(error));
  }
  int w = (int)width;
  int h = (int)height;
  float* data = (float*)malloc(sizeof(float)*w*h);
  int counter = 0;
  for (int i = 0; i < w*h*4; i+=4)
  {
    data[counter] = 0.21*((float)(int)image[i])/255.0;
    data[counter] += 0.72*((float)(int)image[i+1])/255.0;
    data[counter] += 0.07*((float)(int)image[i+2])/255.0;
    counter++;
  }
  Matrix* m = self->matrixUtil->newMatrix(data,w,h);
  Image* ret = self->newImageFromMatrix(self,m);
  return ret;
}

void saveImageToFileImpl(ImageUtil* self, Image* im, char* p)
{
  /*Encode the image*/
  if (VERBOSITY > 0)
  {
      printf("###########    Saving File: ' %s '   ############",p);
  }
  self->matrixUtil->sync(im->pixels);
  const char* path = p;
  unsigned char* saveim = (unsigned char*)malloc(sizeof(unsigned char)*im->shape[0]*im->shape[1]*4);
  int pixelcount = 0;
  float* pix = (float*)im->pixels->nativePtr;
  float mmm = 0.0;
  for (int k=0; k<(im->shape[0] * im->shape[1]); k++)
  {
    if (fabsf(pix[k])> mmm)
    {
      mmm = fabsf(pix[k]);
    }
  }
  if (mmm != 1.0)
  {
    for (int j=0; j<(im->shape[0] * im->shape[1]); j++)
    {
      //printf("[ %f ]",pix[j]);
      pix[j] =fabsf(pix[j])/mmm;
    }
  }
  if (VERBOSITY > 1)
  {
    printf("\nSaved Image Pixel Maximum: %f\n",mmm);
  }
  for (int i = 0; i < im->shape[0]*im->shape[1]*4; i+=4)
  {
    saveim[i] = pix[pixelcount] * 255.0;
    saveim[i+1] = pix[pixelcount] * 255.0;;
    saveim[i+2] = pix[pixelcount] * 255.0;;
    saveim[i+3] = 255;
    pixelcount++;
  }

  unsigned error = lodepng_encode32_file(path, saveim, im->shape[1], im->shape[0]);

  /*if there's an error, display it*/
  if (error)
  {
    printf("error %u: %s\n", error, lodepng_error_text(error));
  }
}

Image* convolveImageImpl(ImageUtil* self, Image* im, Image* kernel)
{
  Matrix* convolvedMat = self->matrixUtil->newEmptyMatrix(im->pixels->shape[0],im->pixels->shape[1]);
  self->matrixUtil->convolve(im->pixels,kernel->pixels,convolvedMat);
  Image* retval = self->newImageFromMatrix(self,convolvedMat);
  return retval;
}

Image* downsampleImageImpl(ImageUtil* self, Image* im, int w, int h)
{
  Matrix* newmat = self->matrixUtil->newEmptyMatrix(w,h);
  self->matrixUtil->downsample(im->pixels,newmat);
  Image* retval = self->newImageFromMatrix(self,newmat);
  return retval;
}

Image* getSSImageAtImpl(ScaleSpaceImage* self, int octave, int scale)
{
  if (octave<self->nOctaves && scale<self->nScalesPerOctave)
  {
    return &(self->scalespace[IDX2C(octave,scale,self->nScalesPerOctave)]);
  } else
  {
    return NULL;
  }
}

void setSSImageAtImpl(ScaleSpaceImage* self, Image* img, int octave, int scale)
{
  if (octave<self->nOctaves && scale<self->nScalesPerOctave)
  {
    self->scalespace[IDX2C(octave,scale,self->nScalesPerOctave)] = *img;
  }
}

void syncSSImageImpl(ImageUtil* self, ScaleSpaceImage* im)
{
  for (int o = 0; o < im->nOctaves; o++)
  {
    for (int s = 0; s < im->nScalesPerOctave; s++)
    {
      Matrix* m =im->getImageAt(im,o,s)->pixels;
      self->matrixUtil->sync(m);
    }
  }
}

ScaleSpaceImage* newEmptySSImageImpl(ImageUtil* self, int octaves, int scalesPerOctave)
{
  ScaleSpaceImage* ssimage = (ScaleSpaceImage*)malloc(sizeof(ScaleSpaceImage));
  ssimage->nScalesPerOctave = scalesPerOctave;
  ssimage->nOctaves = octaves;
  Image* images = (Image*)malloc(sizeof(Image)*octaves*scalesPerOctave);
  ssimage->scalespace = images;
  ssimage->getImageAt = getSSImageAtImpl;
  ssimage->setImageAt = setSSImageAtImpl;
  return ssimage;
}

ScaleSpaceImage* buildPyrmaidImpl(ImageUtil* self, Image* im, int octaves)
{
  ScaleSpaceImage* ssimage = self->newEmptyScaleSpaceImage(self,octaves,1);

  int w = im->shape[0];
  int h = im->shape[1];
  for (int o = 0; o < octaves; o++)
  {
    Image* thisOctave = self->downsample(self,im,w,h);
    ssimage->setImageAt(ssimage,thisOctave,o,1);
    w = w/2;
    h = h/2;
  }
  return ssimage;
}

ImageUtil* GetImageUtil(MatrixUtil* matrixUtil)
{
  ImageUtil* iu = (ImageUtil*)malloc(sizeof(ImageUtil));
  iu->matrixUtil = matrixUtil;
  iu->newEmptyImage = newEmptyImageImpl;
  iu->newImageFromMatrix = newImageFromMatrixImpl;
  iu->newEmptyScaleSpaceImage = newEmptySSImageImpl;
  iu->downsample = downsampleImageImpl;
  iu->buildPyrmaid = buildPyrmaidImpl;
  iu->loadImageFromFile = loadImageFromFileImpl;
  iu->saveImageToFile = saveImageToFileImpl;
  iu->convolve = convolveImageImpl;
  iu->syncScaleSpaceImage = syncSSImageImpl;
  return iu;
}
