#include <stdlib.h>
#include <stdio.h>
#include "utils/PrimitiveMatrixUtil.c"
//#include "structs/PriorityQ.c"

int main(int argc, char const *argv[]) {

  MatrixUtil* matrixUtil = GetPrimitiveMatrixUtil();
  int N = 3;
  float* data = (float*)malloc(sizeof(float)*N*N);
  float* data2 = (float*)malloc(sizeof(float)*N);

  data[0] = 6;
  data[1] = 4;
  data[2] = 3;
  data[3] = 1;
  data[4] = -2;
  data[5] = -2;
  data[6] = 1;
  data[7] = 1;
  data[8] = 1;

  data2[0] = 610;
  data2[1] = 0;
  data2[2] = 120;

  Matrix* X = matrixUtil->newMatrix(data,N,N);
  Matrix* y = matrixUtil->newMatrix(data2,N,1);
  Matrix* solved = matrixUtil->lstsq(X,y);
  matrixUtil->pprint(solved);
  matrixUtil->pprint(X);
  return 0;
}
