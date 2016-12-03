#include "Extractor.h"
#include "Filters.c"
#include "structs/Keypoint.c"

ScaleSpaceImage* BuildGaussianPyramid(ImageUtil* self, Image* im, int octaves, int scalesPerOctave)
{
  ScaleSpaceImage* ssimage = self->newEmptyScaleSpaceImage(self,octaves,scalesPerOctave);
  ScaleSpaceImage* gauss = self->newEmptyScaleSpaceImage(self,1,scalesPerOctave);

  float scale = 1.0/(float)scalesPerOctave;
  for (int i = 0; i < scalesPerOctave; i++)
  {
    float factor = powf(2,scale+(scale*i));
    gauss->setImageAt(gauss,MakeGaussianKernel(self,5,5*factor),0,i);
  }

  int w = im->shape[0];
  int h = im->shape[1];
  for (int o = 0; o < octaves; o++)
  {
    Image* thisOctave = self->downsample(self,im,w,h);
    for (int s = 0; s < scalesPerOctave; s++)
    {
      Image* thisGauss = gauss->getImageAt(gauss,0,s);
      ssimage->setImageAt(ssimage,self->convolve(self,thisOctave,thisGauss),o,s);
    }
    w = w/2;
    h = h/2;
  }
  for (int i = 0; i < scalesPerOctave; i++)
  {
    freeCudaMatrixDeviceMemory(gauss->getImageAt(gauss,0,i)->pixels);
  }
  free(gauss->scalespace);
  free(gauss);
  return ssimage;
}

ScaleSpaceImage* ssDifferenceOfGaussian(ImageUtil* self, ScaleSpaceImage* ssimage)
{
    int octaves = ssimage->nOctaves;
    int scales = ssimage->nScalesPerOctave - 1;
    ScaleSpaceImage* retval = self->newEmptyScaleSpaceImage(self,octaves,scales);

    for (int o = 0; o < octaves; o++)
    {
      for (int s = 0; s < scales; s++)
      {
        Matrix* mat = self->matrixUtil->newEmptyMatrix(ssimage->getImageAt(ssimage,o,s)->shape[0],ssimage->getImageAt(ssimage,o,s)->shape[1]);
        self->matrixUtil->subtract(ssimage->getImageAt(ssimage,o,s)->pixels,  ssimage->getImageAt(ssimage,o,s+1)->pixels, mat);
        self->matrixUtil->pow(mat,2,mat);
        retval->setImageAt(retval,self->newImageFromMatrix(self,mat),o,s);
      }
    }
    return retval;
}
