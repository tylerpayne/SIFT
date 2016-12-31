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
  int gw = 8; // Width of Gaussian Kernels
  int g1s = 5; // Sigma of Gaussian Kernel 1
  int g2s = 3; // Sigma of Gaussian Kernel 2
  int mw = 15; // Width of LocalMax window
  char* saves = "DoG.png"; // Filepath to save to
  path = "image.png"; // Filepath to load from

  for (int i = 1; i < argc; i++)
  {
    switch (i) {
      case 1:
        VERBOSITY = atoi(argv[1]); // Verbosity 0-5
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
  //Load the image
  Image* in = imutil->loadImageFromFile(imutil,path);
  Image* im = imutil->resample(imutil,in,256,256);
  //Create the two gaussians
  Image* gauss1 = MakeGaussianKernel(imutil,gw,g1s);
  Image* gauss2 = MakeGaussianKernel(imutil,gw,g2s);
  //Get difference of gaussian kernel
  Image* DoGKernel = imutil->subtract(imutil,gauss1,gauss2);
  //Convolve
  Image* DoGImage = imutil->convolve(imutil,im,DoGKernel);
  //Find corners (local maximums)
  Image* corners = imutil->max(imutil,DoGImage,mw);
  //Save image
  imutil->saveImageToFile(imutil,im,saves);
  printf("\n\n################################");
  printf("\n################################\n");
  return 0;
}
