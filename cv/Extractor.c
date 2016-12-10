#include "Extractor.h"
#include "Filters.c"
#include "structs/Keypoint.c"

ScaleSpaceImage* BuildGaussianPyramid(ImageUtil* imutil, Image* im, int octaves, int scalesPerOctave)
{
  ScaleSpaceImage* ssimage = imutil->newEmptyScaleSpaceImage(imutil,octaves,scalesPerOctave);
  ScaleSpaceImage* gauss = imutil->newEmptyScaleSpaceImage(imutil,1,scalesPerOctave);

  float scale = 1.0/(float)scalesPerOctave;
  for (int i = 0; i < scalesPerOctave; i++)
  {
    float factor = powf(2,scale+(scale*i));
    gauss->setImageAt(gauss,MakeGaussianKernel(imutil,5,5*factor),0,i);
  }

  int w = im->shape[0];
  int h = im->shape[1];
  for (int o = 0; o < octaves; o++)
  {
    Image* thisOctave = imutil->downsample(imutil,im,w,h);
    for (int s = 0; s < scalesPerOctave; s++)
    {
      Image* thisGauss = gauss->getImageAt(gauss,0,s);
      ssimage->setImageAt(ssimage,imutil->convolve(imutil,thisOctave,thisGauss),o,s);
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

//Image* DifferenceOfGaussian

ScaleSpaceImage* ssDifferenceOfGaussian(ImageUtil* imutil, ScaleSpaceImage* ssimage)
{
    int octaves = ssimage->nOctaves;
    int scales = ssimage->nScalesPerOctave - 1;
    ScaleSpaceImage* retval = imutil->newEmptyScaleSpaceImage(imutil,octaves,scales);

    for (int o = 0; o < octaves; o++)
    {
      for (int s = 0; s < scales; s++)
      {
        Matrix* mat = imutil->matrixUtil->newEmptyMatrix(ssimage->getImageAt(ssimage,o,s)->shape[0],ssimage->getImageAt(ssimage,o,s)->shape[1]);
        imutil->matrixUtil->subtract(ssimage->getImageAt(ssimage,o,s)->pixels,  ssimage->getImageAt(ssimage,o,s+1)->pixels, mat);
        imutil->matrixUtil->pow(mat,2,mat);
        retval->setImageAt(retval,imutil->newImageFromMatrix(imutil,mat),o,s);
      }
    }
    return retval;
}

//Image Derivatives
Image* GetImageDerivativeMagnitude(ImageUtil* imutil, Image* image)
{
  Image* sobel = MakeSobelKernels(imutil);
  Matrix* dx = imutil->convolve(imutil,image,&sobel[0])->pixels;
  Matrix* dy = imutil->convolve(imutil,image,&sobel[1])->pixels;

  MatrixUtil* matutil = imutil->matrixUtil;

  Matrix* retPixels = matutil->newEmptyMatrix(image->shape[0],image->shape[1]);
  matutil->pow(dx,2,dx);
  matutil->pow(dy,2,dy);
  matutil->add(dx,dy,retPixels);
  matutil->sqrt(retPixels,retPixels);

  return imutil->newImageFromMatrix(imutil,retPixels);
}

ScaleSpaceImage* ssGetImageDerivativeMagnitude(ImageUtil* imutil, ScaleSpaceImage* image)
{
  ScaleSpaceImage* retval = imutil->newEmptyScaleSpaceImage(imutil,image->nOctaves,image->nScalesPerOctave);
  for (int o = 0; o < image->nOctaves; o++)
  {
    for (int s = 0; s < image->nScalesPerOctave; s++)
    {
      retval->setImageAt(retval,GetImageDerivativeMagnitude(imutil,image->getImageAt(image,o,s)),o,s);
    }
  }

  return retval;
}

Image* GetImageDerivativeAngle(ImageUtil* imutil, Image* image)
{
  Image* sobel = MakeSobelKernels(imutil);
  Matrix* dx = imutil->convolve(imutil,image,&sobel[0])->pixels;
  Matrix* dy = imutil->convolve(imutil,image,&sobel[1])->pixels;

  MatrixUtil* matutil = imutil->matrixUtil;

  Matrix* retPixels = matutil->newEmptyMatrix(image->shape[0],image->shape[1]);
  matutil->divide(dy,dx,retPixels);
  matutil->arctan(retPixels,retPixels);

  return imutil->newImageFromMatrix(imutil,retPixels);
}

ScaleSpaceImage* ssGetImageDerivativeAngle(ImageUtil* imutil, ScaleSpaceImage* image)
{
  ScaleSpaceImage* retval = imutil->newEmptyScaleSpaceImage(imutil,image->nOctaves,image->nScalesPerOctave);
  for (int o = 0; o < image->nOctaves; o++)
  {
    for (int s = 0; s < image->nScalesPerOctave; s++)
    {
      retval->setImageAt(retval,GetImageDerivativeAngle(imutil,image->getImageAt(image,o,s)),o,s);
    }
  }

  return retval;
}

//Non Maximum Suppression
/*
Image* SuppressNonMax(ImageUtil* imutil, Image* image)
{

}

ScaleSpaceImage* ssSuprressNonMax(ImageUtil* imutil, ScaleSpaceImage* image)
{

}*/

//LocateExtrema
  //FindMaximums

//Extract Keypoints

//Assign Keypoint Orientations

//Make SIFT Descriptors
