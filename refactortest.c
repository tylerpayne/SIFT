#include <utils/MatrixUtil.h>
#include <utils/ImageUtil.h>
#include <structs/Keypoint.c>
#include <cv/Extractor.c>
#include <cv/Matcher.c>
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
  Matcher* matcher = NewMatcher(imutil);
  IOUtil* ioutil = GetIOUtil(imutil);
  DrawUtil* drawutil = GetDrawUtil();

  Image* l_image = ioutil->loadImageFromFile(ioutil,"lena.png");
  Array* l_points = extractor->findCornerKeypoints(extractor,l_image,9,9,5,15,NULL);
  Matrix* l_features = extractor->makeFeatureMatrixFromKeypointDescriptors(extractor,l_points);
  Image* l_featim = imutil->newImageFromMatrix(imutil,l_features);
  ioutil->saveImageToFile(ioutil,l_featim,ioutil->appendNumberToFilename("test",100),PNG);
/*
  Image* r_image = ioutil->loadImageFromFile(ioutil,"right015.png");
  Array* r_points = extractor->findCornerKeypoints(extractor,r_image,15,5,3,9,NULL);
  Matrix* r_features = extractor->makeFeatureMatrixFromKeypointDescriptors(extractor,r_points);
  Image* r_featim = imutil->newImageFromMatrix(imutil,r_features);
  ioutil->saveImageToFile(ioutil,r_featim,"r_featurematriximage.png",PNG);

  Image* matches = matcher->findMatches(matcher,l_features,r_features,l_points,r_points);
  ioutil->saveImageToFile(ioutil,matches,"matches.png",PNG);

  int nWindowWidth = 100;
  int radius = 50;
  int saved = 0;
  for (int i = 0; i < l_points->count; i++)
  {
    Keypoint* kp = ((Keypoint**)l_points->ptr)[i];
    void* hasMatch = kp->get(kp,"hasMatch");

    if (hasMatch != NULL)
    {
      if (((int*)hasMatch)[0] == 1)
      {
        Keypoint* rkp = (Keypoint*)(kp->get(kp,"match"));

        Matrix* zPixels = matutil->newEmptyMatrix(nWindowWidth,nWindowWidth);
        Rect size = {nWindowWidth,nWindowWidth};
        Point2 Aidx = {max(0,(int)(kp->position[0])-radius),max(0,(int)(kp->position[1])-radius)};
        Point2 Bidx = {0,0};
        matutil->copy(matutil,kp->sourceImage->pixels,zPixels,size,Aidx,Bidx);
        Image* lookPatch = imutil->newImageFromMatrix(imutil,zPixels);
        Matrix* fPixels = matutil->newEmptyMatrix(nWindowWidth,nWindowWidth);
        //size = {nWindowWidth,nWindowWidth};
        Point2 r_Aidx = {max(0,(int)(rkp->position[0])-radius),max(0,(int)(rkp->position[1])-radius)};
        Point2 r_Bidx = {0,0};
        matutil->copy(matutil,rkp->sourceImage->pixels,fPixels,size,r_Aidx,r_Bidx);
        Image* foundPatch = imutil->newImageFromMatrix(imutil,fPixels);
        char* png = ".png";
        char* s = "ls";
        char* f = "lf";
        char* id;
        int offset;
        if (saved < 10)
        {
          id = (char*)malloc(sizeof(char));
          id[0] = saved + '0';
          offset = 1;
        } else if (saved < 100)
        {
          id = (char*)malloc(sizeof(char)*2);
          id[0] = ((saved/10)%10) + '0';
          id[1] = (saved%10) + '0';
          offset = 2;
        } else if (saved <1000)
        {
          id = (char*)malloc(sizeof(char)*3);
          id[0] = ((saved/100)%100) + '0';
          id[1] = (saved%100) + '0';
          id[2] = (saved%10) + '0';
          offset =3;
        }
        char* searchString = (char*)malloc(sizeof(char)*(7+offset));
        memcpy(&searchString[0],id,sizeof(char)*offset);
        searchString[0+offset] = s[0];
        searchString[1+offset] = s[1];
        memcpy(&searchString[2+offset],png,sizeof(char)*5);

        char* foundString = (char*)malloc(sizeof(char)*(7+offset));
        memcpy(&foundString[0],id,sizeof(char)*offset);
        foundString[0+offset] = f[0];
        foundString[1+offset] = f[1];
        memcpy(&foundString[2+offset],png,sizeof(char)*5);
        saved++;
        ioutil->saveImageToFile(ioutil,lookPatch,searchString,PNG);
        ioutil->saveImageToFile(ioutil,foundPatch,foundString,PNG);
      }
    }
  }

  //imutil->subPixelAlignImageIndexPair(imutil,corners);
  //ioutil->saveImageToFile(ioutil,DoGImage,"gdklena.png",PNG);
  */
  drawutil->drawKeypoints(drawutil,l_points,l_image,NULL);
  gtk_main();
  cudaDeviceReset();
  return 0;
}
