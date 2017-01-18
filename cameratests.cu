#include "structs/Core.h"
#include "utils/CUDAImageUtil.cu"
#include "structs/BinaryTreeNode.c"
#include "structs/Keypoint.c"
#include "structs/KDTree.c"
#include "cv/Extractor.c"
#include "cv/Matcher.c"
#include "world/Camera.c"

int main(int argc, char const *argv[])
{
  //int FEATURE = 0;
  float CONTRASTTHRESHOLD = 0.1;
  int BINS = 8;
  int WINDOWSIZE = 9;
  for (int i  = 0; i < argc; i++)
  {
    switch (i) {
      case 1:
        VERBOSITY = atoi(argv[1]);
        break;
      case 2:
        WINDOWSIZE = atoi(argv[2]);
        break;
      case 3:
        CONTRASTTHRESHOLD = atof(argv[3]);
        break;
      case 4:
        BINS = atoi(argv[4]);
        break;
    };
  }


  //UTILS
  MatrixUtil* matutil = GetCUDAMatrixUtil(1);
  ImageUtil* imutil = GetCUDAImageUtil(matutil);
  Extractor* extractor = NewExtractor(imutil);
  Matcher* matcher = NewMatcher(imutil);
  printf("UTILS READY\n");

  //CAMERAS
  float mmFocalLength = 29;
  float mmSensorWidth = 4.8;
  float origin[] = {0,0,0};
  float rotation[] = {0,0,0};
  Matrix* pxyz = matutil->newMatrix(&origin[0],3,1);
  Matrix* rxyz = matutil->newMatrix(&rotation[0],3,1);

  Image* leftFrame = imutil->loadImageFromFile(imutil,"left015.png");
  Image* rightFrame = imutil->loadImageFromFile(imutil,"right015.png");

  /*Image* gauss = extractor->filters->makeGaussianKernel(extractor->filters,5,5);

  Image* leftFrame = imutil->convolve(imutil,leftFrameraw,gauss);
  Image* rightFrame = imutil->convolve(imutil,rightFrameraw,gauss);
  leftFrameraw->free(leftFrameraw);
  rightFrameraw->free(rightFrameraw);
  gauss->free(gauss);*/
//LEFT
  Camera* LEFTCAM = NewCamera(matutil,NULL,NULL,leftFrame);
  LEFTCAM->setIntrinsics(LEFTCAM,mmFocalLength,mmSensorWidth,leftFrame->shape.width,leftFrame->shape.height);

  LEFTCAM->setPosition(LEFTCAM,pxyz);
  LEFTCAM->setRotation(LEFTCAM,rxyz);
//RIGHT
  Camera* RIGHTCAM = NewCamera(matutil,NULL,NULL,rightFrame);
  RIGHTCAM->setIntrinsics(RIGHTCAM,mmFocalLength,mmSensorWidth,leftFrame->shape.width,leftFrame->shape.height);

  RIGHTCAM->setPosition(RIGHTCAM,pxyz);
  RIGHTCAM->setRotation(RIGHTCAM,rxyz);
  printf("CAMERAS READY\n");
  //FEATURES
  Array* leftCorners = extractor->findCornerKeypoints(extractor,leftFrame,9,5,3,WINDOWSIZE,&CONTRASTTHRESHOLD);
  Array* rightCorners = extractor->findCornerKeypoints(extractor,rightFrame,9,5,3,WINDOWSIZE,&CONTRASTTHRESHOLD);
  printf("GOT %i, %i CORNERS\n",leftCorners->count, rightCorners->count);

  ImageGradientVectorPair* lg = imutil->gradients(imutil,leftFrame);
  ImageGradientVectorPair* rg = imutil->gradients(imutil,rightFrame);
  Image* leftGrad = imutil->multiply(imutil,lg->magnitude,lg->angle);
  Image* rightGrad = imutil->multiply(imutil,rg->magnitude,rg->angle);
  printf("GOT GRADIENTS %i %i\n",leftGrad==NULL,rightGrad==NULL);
  extractor->makeFeatureDescriptorsAtKeypoints(extractor,leftCorners,leftGrad,16);
  extractor->makeFeatureDescriptorsAtKeypoints(extractor,rightCorners,rightGrad,16);
  printf("madekeypoints\n");
  Matrix* leftFeatMatrix = extractor->makeFeatureMatrixFromKeypointDescriptors(extractor,leftCorners);
  Matrix* rightFeatMatrix = extractor->makeFeatureMatrixFromKeypointDescriptors(extractor,rightCorners);
  printf("madefeaturematrix\n");
  //extractor->unorientFeatureMatrix(extractor,leftFeatMatrix,36);
  //extractor->unorientFeatureMatrix(extractor,rightFeatMatrix,36);
  Matrix* leftGenFeatures = extractor->generalizeFeatureMatrix(extractor,leftFeatMatrix,BINS);
  Matrix* rightGenFeatures = extractor->generalizeFeatureMatrix(extractor,rightFeatMatrix,BINS);
  leftFeatMatrix->free(leftFeatMatrix);
  rightFeatMatrix->free(rightFeatMatrix);
  printf("EXTRACTED FEATURES!\n");
  Image* leftfeat = imutil->newImageFromMatrix(imutil,leftGenFeatures);
  imutil->saveImageToFile(imutil,leftfeat,"leftfeaturesmatrix.png");
  Image* rightfeat = imutil->newImageFromMatrix(imutil,rightGenFeatures);
  imutil->saveImageToFile(imutil,rightfeat,"rightfeaturesmatrix.png");

  matcher->findMatches(matcher,leftGenFeatures,rightGenFeatures,leftCorners,rightCorners);
  int nWindowWidth = 100;
  int radius = 50;
  int saved = 0;
  for (int i = 0; i < leftCorners->count; i++)
  {
    Keypoint* kp = ((Keypoint**)leftCorners->ptr)[i];
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
        size = {nWindowWidth,nWindowWidth};
        Aidx = {max(0,(int)(rkp->position[0])-radius),max(0,(int)(rkp->position[1])-radius)};
        Bidx = {0,0};
        matutil->copy(matutil,rkp->sourceImage->pixels,fPixels,size,Aidx,Bidx);
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
        imutil->saveImageToFile(imutil,lookPatch,searchString);
        imutil->saveImageToFile(imutil,foundPatch,foundString);
      }
    }
  }
/*
  saved = 0;
  for (int i = 0; i < rightCorners->count; i++)
  {
    Keypoint* kp = ((Keypoint**)rightCorners->ptr)[i];
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
        size = {nWindowWidth,nWindowWidth};
        Aidx = {max(0,(int)(rkp->position[0])-radius),max(0,(int)(rkp->position[1])-radius)};
        Bidx = {0,0};
        matutil->copy(matutil,rkp->sourceImage->pixels,fPixels,size,Aidx,Bidx);
        Image* foundPatch = imutil->newImageFromMatrix(imutil,fPixels);
        char* png = ".png\0";
        char* s = "rs";
        char* f = "rf";
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
        searchString[0] = s[0];
        searchString[1] = s[1];
        memcpy(&searchString[2],id,sizeof(char)*offset);
        memcpy(&searchString[2+offset],png,sizeof(char)*5);

        char* foundString = (char*)malloc(sizeof(char)*(7+offset));
        foundString[0] = f[0];
        foundString[1] = f[1];
        memcpy(&foundString[2],id,sizeof(char)*offset);
        memcpy(&foundString[2+offset],png,sizeof(char)*5);

        saved++;

        imutil->saveImageToFile(imutil,lookPatch,searchString);
        imutil->saveImageToFile(imutil,foundPatch,foundString);
      }
    }
  }
  //imutil->saveImageToFile(imutil,imMatches,"MATCHES.png");
/*
//KDTREE
  KDTree* tree = NewKDTree((Keypoint**)(leftCorners.ptr),leftCorners.count,19);
  printf("BUILT KD TREE. DEPTH: %i\n",tree->height);
  int z = rand()%(rightCorners.count-1);
  Keypoint* zp = ((Keypoint**)rightCorners.ptr)[z];
  Matrix* lookVector = (Matrix*)(zp->get(zp,"feature"));
  printf("SEARCHING! (%f,%f)\n",(float)(zp->position[0]),(float)(zp->position[1]));
  TreeNode* nextBest = (TreeNode*)malloc(sizeof(TreeNode));
  TreeNode* found = tree->search(tree,lookVector,nextBest,1);
  Keypoint* foundKey = (Keypoint*)(found->value);
  Keypoint* nextBestKey = (Keypoint*)(nextBest->value);
  //float confidence;
  printf("FOUND (%f,%f)\n",(float)(foundKey->position[0]),(float)(foundKey->position[1]));

  int nWindowWidth = 100;
  int radius = nWindowWidth/2;

  Matrix* zPixels = matutil->newEmptyMatrix(nWindowWidth,nWindowWidth);
  Rect size = {nWindowWidth,nWindowWidth};
  Point2 Aidx = {(int)(zp->position[0])-radius,(int)(zp->position[1])-radius};
  Point2 Bidx = {0,0};
  matutil->copy(matutil,zp->sourceImage->pixels,zPixels,size,Aidx,Bidx);
  Image* lookPatch = imutil->newImageFromMatrix(imutil,zPixels);

  Matrix* fPixels = matutil->newEmptyMatrix(nWindowWidth,nWindowWidth);
  size = {nWindowWidth,nWindowWidth};
  Aidx = {(int)(foundKey->position[0])-radius,(int)(foundKey->position[1])-radius};
  Bidx = {0,0};
  matutil->copy(matutil,foundKey->sourceImage->pixels,fPixels,size,Aidx,Bidx);
  Image* foundPatch = imutil->newImageFromMatrix(imutil,fPixels);

  Matrix* nnPixels = matutil->newEmptyMatrix(nWindowWidth,nWindowWidth);
  size = {nWindowWidth,nWindowWidth};
  Aidx = {(int)(nextBestKey->position[0])-radius,(int)(nextBestKey->position[1])-radius};
  Bidx = {0,0};
  matutil->copy(matutil,nextBestKey->sourceImage->pixels,nnPixels,size,Aidx,Bidx);
  Image* nnPatch = imutil->newImageFromMatrix(imutil,nnPixels);

  imutil->saveImageToFile(imutil,lookPatch,"Search.png");
  imutil->saveImageToFile(imutil,foundPatch,"Found.png");
  imutil->saveImageToFile(imutil,nnPatch,"FoundNN1.png");*/

  cudaDeviceReset();

  return 0;
}
