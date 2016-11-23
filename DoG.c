#include <stdlib.h>
#include <stdio.h>
#include "utils/PrimitiveMatrixUtil.c"
#include "utils/ImageUtil.c"
//#include "structs/PriorityQ.c"

int main(int argc, char const *argv[])
{
  printf("\n################################\n\n");
  for (int i = 0; i < argc; i ++)
  {
    printf("Argument %i: %s\n",i,argv[i]);
  }
  printf("\n################################\n\n");
  char* path;
  int gw;
  float gs;
  int g2w;
  float g2s;
  if (argc != 6)
  {
    path = "image.png";
    gw = 15;
    gs = 15;
    g2w = 15;
    g2s = 5;
  } else
  {
    path = (char*)argv[1];
    gw = atoi(argv[2]);
    gs = atof(argv[3]);
    g2w = atoi(argv[4]);
    g2s = atof(argv[5]);
  }

  printf("Path: %s\n",path);
  printf("Values: %i, %f, %i, %f\n", gw,gs,g2w,g2s);

  printf("\n################################\n\n");

  MatrixUtil* matutil = GetPrimitiveMatrixUtil();
  ImageUtil* imutil = GetImageUtil(matutil);

  Image* image = imutil->loadImageFromFile(imutil,path);
  Image* gauss = imutil->generateGaussian(imutil,gw,gs);
  Image* gauss2 = imutil->generateGaussian(imutil,g2w,g2s);
  //matutil->pprint(gauss->pixels,"Gauss");
  Image* blur = imutil->convolve(imutil,image,gauss);
  Image* blur2 = imutil->convolve(imutil,image,gauss2);
  Image* dog = imutil->newEmptyImage(imutil,blur->shape[0],blur->shape[1]);
  dog->pixels = matutil->pow(matutil->subtract(blur->pixels,blur2->pixels),2.0);
  imutil->saveImageToFile(imutil,dog,"DoG.png");
  return 0;
}
