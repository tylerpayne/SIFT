#include <utils/MatrixUtil.h>
#include <utils/ImageUtil.h>
#include <structs/Keypoint.c>
#include <cv/Extractor.c>
#include <utils/IOUtil.c>
#include <utils/DrawUtil.c>

int main(int argc, char const *argv[]) {
  char** gargv = argv;
  gtk_init(&argc,&gargv);

  cudaDeviceReset();

  if (argc==2)
  {
    VERBOSITY = atoi(argv[1]);
  }

  MatrixUtil* matutil = GetMatrixUtil();
  ImageUtil* imutil = GetImageUtil(matutil);
  Extractor* extractor = NewExtractor(imutil);
  IOUtil* ioutil = GetIOUtil(imutil);
  DrawUtil* drawutil = GetDrawUtil();

  Image* image = ioutil->loadImageFromFile(ioutil,"lena.png");
  Array* points = extractor->findCornerKeypoints(extractor,image,15,13,9,20,NULL);
  Matrix* featMatrix = extractor->makeFeatureMatrixFromKeypointDescriptors(extractor,points);
  printf("featMatrix: %i",featMatrix==NULL);
  Image* featim = imutil->newImageFromMatrix(imutil,featMatrix);
  ioutil->saveImageToFile(ioutil,featim,"featurematrix.png",PNG);
  //imutil->subPixelAlignImageIndexPair(imutil,corners);
  //ioutil->saveImageToFile(ioutil,DoGImage,"gdklena.png",PNG);
  drawutil->drawKeypoints(drawutil,points,image,NULL);
  gtk_main();
  return 0;
}
