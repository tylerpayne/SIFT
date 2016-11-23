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
      printf("###########   Saving File: ' %s '   ############",p);
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

/*ScaleSpaceImage* buildGaussianPyrmaid(Image* im, int octaves, int scalesperoctave)
{

}*/

Image* generateGaussianImpl(ImageUtil* self, int width, float std)
{
  float* data = (float*)malloc(sizeof(float)*width*width);
  int radius = width/2;
  float variance = std*std;
  //float coeff = 1.0/(2*M_PI);
  float sum = 0;
  for (int j = 0; j < width; j++)
  {
    for (int i = 0; i < width; i++)
    {
      float x = j-radius;
      float y = i-radius;
      float power = -1.0*(((x*x)+(y*y))/(2*variance));
      float val = exp(power);
      data[IDX2C(i,j,width)] = val;
      sum += val;
      //printf("[ %f ]",data[IDX2C(i,j,width)]);
    }
    //printf("\n");
  }
  for (int r = 0; r < width*width; r++)
  {
    data[r] = data[r]/sum;
  }
  Matrix* m = self->matrixUtil->newMatrix(data,width,width);
  return self->newImageFromMatrix(self,m);
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
  return &self->scalespace[IDX2C(octave,scale,self->nScalesPerOctave)];
}

ScaleSpaceImage* buildPyrmaidImpl(ImageUtil* self, Image* im, int octaves, int scalesPerOctave)
{
  ScaleSpaceImage* ssimage = (ScaleSpaceImage*)malloc(sizeof(ScaleSpaceImage));
  ssimage->nScalesPerOctave = scalesPerOctave;
  ssimage->nOctaves = octaves;
  Image* images = (Image*)malloc(sizeof(Image)*octaves*scalesPerOctave);

  int w = im->shape[0];
  int h = im->shape[1];

  for (int o = 0; o < octaves; o++)
  {
    float scale = 1.0/(float)scalesPerOctave;
    Image* thisOctave = self->downsample(self,im,w,h);
    for (int s = 0; s < scalesPerOctave; s++)
    {
      float factor = powf(2,scale+(scale*s));
      Image* gauss = self->generateGaussian(self,10,5*factor);
      images[IDX2C(o,s,scalesPerOctave)] = *self->convolve(self,thisOctave,gauss);
      freeCudaMatrixDeviceMemory(gauss->pixels);
      free(gauss);
    }
    w = w/2;
    h = h/2;
  }

  ssimage->scalespace = images;
  ssimage->getImageAt = getSSImageAtImpl;
  return ssimage;
}

ImageUtil* GetImageUtil(MatrixUtil* matrixUtil)
{
  ImageUtil* iu = (ImageUtil*)malloc(sizeof(ImageUtil));
  iu->matrixUtil = matrixUtil;
  iu->newEmptyImage = newEmptyImageImpl;
  iu->newImageFromMatrix = newImageFromMatrixImpl;
  iu->downsample = downsampleImageImpl;
  iu->buildPyrmaid = buildPyrmaidImpl;
  iu->loadImageFromFile = loadImageFromFileImpl;
  iu->saveImageToFile = saveImageToFileImpl;
  iu->generateGaussian = generateGaussianImpl;
  iu->convolve = convolveImageImpl;
  return iu;
}
