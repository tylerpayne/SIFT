#include <stdlib.h>
#include <stdio.h>
#include "utils/CUDAMatrixUtil.cu"
//#include "cv/Extractor.c"

int main(int argc, char const *argv[]) {
  printf("\n################################");
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

  MatrixUtil* matutil = GetCUDAMatrixUtil(1);

  int N = 3;
  int M = 3;
  int K = 3;

  float* a  = (float*)malloc(sizeof(float)*N*K);
  float* b  = (float*)malloc(sizeof(float)*K*M);
  float* c = (float*)malloc(sizeof(float)*N*M);

  for (int z = 0 ;z < N*K;z++)
  {
    a[z] = 1;

  }
  for (int z = 0 ;z < K*M;z++)
  {
    b[z] = 1;

  }

  /*Matrix* A = matutil->newMatrix(b,1,3);
  Matrix* B = matutil->newMatrix(b,1,3);
  Matrix* C = matutil->newEmptyMatrix(1,3);
  matutil->pprint(C,"PreAdd");
  matutil->add(A,B,C);
  matutil->pprint(C,"Add");*/
  Matrix* A = matutil->newMatrix(a,N,M);
  //matutil->pprint(matutil,A,"A");
  Matrix* B = matutil->newMatrix(b,N,M);
  Matrix* C = matutil->newEmptyMatrix(N,M);
  cudaDeviceSynchronize();
  matutil->pprint(matutil,A,"A");
  matutil->pprint(matutil,B,"B");
  matutil->dot(matutil,A,B,C);
  cudaDeviceSynchronize();
  matutil->pprint(matutil,C,"C");
  printf("\n\n################################");
  printf("\n################################");
  return 0;
}
