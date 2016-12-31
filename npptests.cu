#include <nppi.h>
#include <string.h>
#include "cv/Filters.c"
#include <cuda_profiler_api.h>

int main(int argc, char const *argv[]) {
  printf("\n################################");
  printf("\n################################\n\n");
  for (int i = 0; i < argc; i ++)
  {
    printf("Argument %i: %s\n",i,argv[i]);
  }
  printf("\n################################\n\n");
  char* path;
  int gw = 8;
  int g1s = 5;
  int g2s = 3;
  int mw = 15;
  char* saves = "npptests.png";
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
        gw = atoi(argv[3]);
        break;
      case 4:
        g1s = atoi(argv[4]);
        break;
      case 5:
        g2s = atoi(argv[5]);
        break;
      case 6:
        mw = atoi(argv[6]);
        break;
      case 7:
        saves = (char*)argv[7];
        break;
    }
  }

  printf("Path: %s\n\n",path);
  ImageUtil* imutil = GetImageUtil(1);

  Image* im = imutil->loadImageFromFile(imutil,path);

  Image* ret = imutil->gradientAngle(imutil,im);

  imutil->saveImageToFile(imutil,ret,saves);
  printf("\n\n################################");
  printf("\n################################\n");
  return 0;
}
