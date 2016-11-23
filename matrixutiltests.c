#include <stdlib.h>
#include <stdio.h>
#include "utils/PrimitiveMatrixUtil.c"
//#include "structs/PriorityQ.c"

int main(int argc, char const *argv[]) {

  MatrixUtil* matrixUtil = GetPrimitiveMatrixUtil();
  int N =3;
  float* xdata = (float*)malloc(sizeof(float)*N*N);
  float* ydata = (float*)malloc(sizeof(float)*N);

  xdata[0] = 1;
  xdata[1] = 2;
  xdata[2] = 2;
  xdata[3] = 3;
  xdata[4] = 4;
  xdata[5] = 5;
  xdata[6] = 6;
  xdata[7] = 7;
  xdata[8] = 8;

  ydata[0] = 10;
  ydata[1] = -25;
  ydata[2] = 35;

  Matrix* X = matrixUtil->newMatrix(xdata,N,N);
  Matrix* y = matrixUtil->newMatrix(ydata,N,N);

  Matrix* solved = matrixUtil->lstsq(X,y);

  matrixUtil->pprint(solved,"Linear Least Squares");

  return 0;
}
