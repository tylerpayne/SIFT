#include <stdlib.h>
#include <stdio.h>
#include "utils/CUDAMatrixUtil.cu"
#include "cv/filters.c"


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
  int saveo = 3;
  int saves = 3;
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

  Image* sobel = MakeSobelKernels(imutil);

  Image* dx = imutil->convolve(imutil,image,&sobel[0]);
  Image* dy = imutil->convolve(imutil,image,&sobel[1]);

  Matrix* m = matutil->newEmptyMatrix(image->shape[0],image->shape[1]);
  matutil->divide(dy->pixels,dx->pixels,m);
  matutil->arctan(m,m);

  //matutil->pow(dx->pixels,2,dx->pixels);
  //matutil->pow(dy->pixels,2,dy->pixels);
  //matutil->add(dy->pixels,dx->pixels,m);
  //matutil->sqrt(m,m);

  Image* gauss = imutil->generateGaussian(imutil,15,15);

  Image* saveim = imutil->convolve(imutil,imutil->newImageFromMatrix(imutil,m),gauss);

  imutil->saveImageToFile(imutil,saveim,"rotderiv.png");
  return 0;
}
