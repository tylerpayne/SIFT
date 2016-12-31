#include <nppi.h>
#include <string.h>
#include "cv/Filters.c"
#include <cuda_profiler_api.h>

int main(int argc, char const *argv[]) {
  printf("\n################################");
  printf("\n################################\n\n");
  char* saves = "npptests.png";
  char* path = "image.png";

  for (int i = 1; i < argc; i++)
  {
    switch (i) {
      case 1:
        VERBOSITY = atoi(argv[1]);
        printf("VERBOSITY = %i\n",VERBOSITY);
        break;
      case 2:
        path = (char*)argv[2];
        printf("FILEPATH = %s\n",path);
        break;
      case 3:
        saves = (char*)argv[3];
        printf("SAVE_FILEPATH = %s\n",saves);
        break;
    }
  }

  ImageUtil* imutil = GetImageUtil(1);

  Image* im = imutil->loadImageFromFile(imutil,path);

  Image* ret = imutil->gradientMagnitude(imutil,im);

  imutil->saveImageToFile(imutil,ret,saves);
  printf("\n\n################################");
  printf("\n################################\n");
  return 0;
}
