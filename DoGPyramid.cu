#include <stdlib.h>
#include <stdio.h>
#include "utils/CUDAMatrixUtil.cu"
#include "cv/Extractor.c"
#include <cuda_profiler_api.h>


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
  cudaProfilerStart();
  MatrixUtil* matutil = GetCUDAMatrixUtil();
  ImageUtil* imutil = GetImageUtil(matutil);

  Image* image = imutil->loadImageFromFile(imutil,path);

  ScaleSpaceImage* pyramid = BuildGaussianPyramid(imutil,image,8,5);
  ScaleSpaceImage* DoG = ssDifferenceOfGaussian(imutil,pyramid);

  imutil->saveImageToFile(imutil,DoG->getImageAt(DoG,3,3),"scalespace.png");
  cudaProfilerStop();
  return 0;
}
