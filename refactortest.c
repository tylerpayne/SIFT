#include <utils/MatrixUtil.h>
#include <utils/ImageUtil.h>
#include <structs/Keypoint.h>
#include <generators/Filters.h>
#include <operators/Extractor.h>
#include <utils/IOUtil.h>
#include <utils/DrawUtil.h>

int main(int argc, char const *argv[]) {
  char** gargv = argv;
  gtk_init(&argc,&gargv);

  if (argc==2)
  {
    VERBOSITY = atoi(argv[1]);
  }

  MatrixUtil* matutil = GetMatrixUtil();
  ImageUtil* imutil = GetImageUtil(matutil);
  Extractor* extractor = NewExtractor(imutil);
  IOUtil* ioutil = GetIOUtil(imutil);
  DrawUtil* drawutil = GetDrawUtil();

  Image* l_image = ioutil->loadImageFromFile(ioutil,"lena.png");
  printf("%s\n","IMLOAD");
  Image* gauss1 = extractor->filters->makeGaussianKernel(extractor->filters,15,15);
  Image* gauss2 = extractor->filters->makeGaussianKernel(extractor->filters,15,9);
  printf("%s\n","makegauss");
  Image* DoGKernel = extractor->imutil->subtract(extractor->imutil,gauss1,gauss2);
  printf("%s\n","sub");

  gauss1->free(gauss1);
  gauss2->free(gauss2);
  printf("%s\n","free");

  Image* DoGImage = extractor->imutil->convolve(extractor->imutil,l_image,DoGKernel);
  printf("%s\n","convolve");

  DoGKernel->free(DoGKernel);
  printf("%s\n","free");

  ImageIndexPair* corners = extractor->imutil->maxIdx(extractor->imutil,DoGImage,50);
  printf("%s\n","maxidx");

  imutil->subPixelAlignImageIndexPair(imutil,corners);
  printf("%s\n","subpix");
  cudaDeviceReset();
  return 0;
}
