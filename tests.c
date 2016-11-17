#include <stdlib.h>
#include <stdio.h>
#include "utils/PrimitiveMatrixUtil.c"
#include "structs/PriorityQ.c"

int main(int argc, char const *argv[]) {

  MatrixUtil* matrixUtil = GetPrimitiveMatrixUtil();
  int N = 100;
  float* data = (float*)malloc(sizeof(float)*N*N);
  float* data2 = (float*)malloc(sizeof(float)*N*N);
  for (int i=0;i<N*N;i++)
  {
    data[i] = 2;
    data2[i] = 4;
  }
  Matrix* m = matrixUtil->newMatrix(data,N,N);
  Matrix* n = matrixUtil->newMatrix(data2,N,N);
  Matrix* k = matrixUti->dot(m,n);
  printf("%f %f\n%f %f",k->getElement(k,0,0),k->getElement(k,0,N-1),k->getElement(k,N-1,0),k->getElement(k,N-1,N-1));
  return 0;
}
