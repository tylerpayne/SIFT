#include <stdlib.h>
#include <stdio.h>
#include "utils/CUDAMatrixUtil.cu"
#include "utils/ImageUtil.cu"

int main(int argc, char const *argv[]) {

  printf("\n################################\n\n");
  for (int i = 0; i < argc; i ++)
  {
    printf("Argument %i: %s\n",i,argv[i]);
  }
  printf("\n################################\n\n");
  char* path;
  int octaves = 8;
  int scales = 5;
  int saveo = 2;
  int saves = 2;
  path = "image.png";

  for (int i = 1; i < argc; i++)
  {
    switch (i) {
      case 1:
        VERBOSITY = atoi(argv[1]);
        break;
      case 2:
        path = (char*)argv[2];
        break;
      case 3:
        octaves = atoi(argv[3]);
        break;
      case 4:
        scales = atoi(argv[4]);
        break;
      case 5:
        saveo = atoi(argv[5]);
        break;
      case 6:
        saves = atoi(argv[6]);
        break;
    }
  }

  printf("Path: %s\n",path);
  printf("Values: %i, %i, (%i,%i)\n", octaves,scales,saveo,saves);

  printf("\n################################\n\n");

  MatrixUtil* matutil = GetCUDAMatrixUtil();
  ImageUtil* imutil = GetImageUtil(matutil);

  Image* image = imutil->loadImageFromFile(imutil,path);
  ScaleSpaceImage* newim = imutil->buildPyrmaid(imutil,image,octaves,scales);
  /*Image* gauss = imutil->generateGaussian(imutil,gw,gs);
  matutil->pprint(gauss->pixels,"Gauss");
  Image* gauss2 = imutil->generateGaussian(imutil,g2w,g2s);
  //matutil->pprint(gauss->pixels,"Gauss");
  //Image* saveim = imutil->newEmptyImage(imutil,image->shape[0],image->shape[1]);
  //matutil->convolve(image->pixels,gauss->pixels,image->pixels);
  //Image* saveim = imutil->convolve(imutil,image,gauss);
  Image* blur = imutil->convolve(imutil,image,gauss);
  Image* blur2 = imutil->convolve(imutil,image,gauss2);
  Image* dog = imutil->newEmptyImage(imutil,blur->shape[0],blur->shape[1]);
  matutil->subtract(blur->pixels,blur2->pixels,dog->pixels);
  //Image* saveim = imutil->newEmptyImage(imutil,dog->shape[0],dog->shape[1]);
  matutil->pow(dog->pixels,2.0,dog->pixels);*/
  imutil->saveImageToFile(imutil,newim->getImageAt(newim,saveo,saves),"pyramid.png");

  return 0;
}
