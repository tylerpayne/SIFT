#include <MatrixUtil.h>
#include <ImageUtil.h>

int main(int argc, char const *argv[]) {

  MatrixUtil* matutil = GetMatrixUtil();
  ImageUtil* imutil = GetImageUtil(matutil);

  return 0;
}
