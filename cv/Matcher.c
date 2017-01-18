#include "Matcher.h"

void findMatchesImpl(Matcher* self, Matrix* feat1, Matrix* feat2, Array* kp1, Array* kp2)
{
  Matrix* sqrErr = self->matutil->newEmptyMatrix(feat1->shape[0],feat2->shape[0]);
  self->matutil->featureDistance(self->matutil,feat1,feat2,sqrErr);
  Image* sqrim = self->imutil->newImageFromMatrix(self->imutil,sqrErr);
  self->imutil->saveImageToFile(self->imutil,sqrim,"matches.png");
  int* matchesPerFeature1 = self->matutil->minRows(self->matutil,sqrErr);
  Matrix* TsqrErr = self->matutil->newEmptyMatrix(sqrErr->shape[1],sqrErr->shape[0]);
  self->matutil->transpose(self->matutil,sqrErr,TsqrErr);
  int* matchesPerFeature2 = self->matutil->minRows(self->matutil,TsqrErr);
  for (int i = 0; i < sqrErr->shape[0]; i++)
  {
    int lkpId,rkpId;
    lkpId = i;
    rkpId = matchesPerFeature1[i];
    if (matchesPerFeature2[rkpId] == i)
    {
      Keypoint* lkp = ((Keypoint**)kp1->ptr)[lkpId];
      Keypoint* rkp = ((Keypoint**)kp2->ptr)[rkpId];
      int* hasMatch = (int*)malloc(sizeof(int));
      hasMatch[0] = 1;
      lkp->set(lkp,"hasMatch",(void*)&hasMatch[0]);
      rkp->set(rkp,"hasMatch",(void*)&hasMatch[0]);
      lkp->set(lkp,"match",(void*)rkp);
      rkp->set(rkp,"match",(void*)lkp);
    }
  }

/*
  for (int i = 0; i < sqrErr->shape[0]; i++)
  {
    int lkpId,rkpId;
    rkpId = i;
    lkpId = matchesPerFeature2[i];

    Keypoint* lkp = ((Keypoint**)kp1->ptr)[lkpId];
    Keypoint* rkp = ((Keypoint**)kp2->ptr)[rkpId];
    int* hasMatch = (int*)malloc(sizeof(int));
    hasMatch[0] = 1;
    rkp->set(rkp,"hasMatch",(void*)&hasMatch[0]);

    rkp->set(rkp,"match",(void*)lkp);
  }*/

}

Matcher* NewMatcher(ImageUtil* imutil)
{
  Matcher* self = (Matcher*)malloc(sizeof(Matcher));
  self->imutil = imutil;
  self->matutil = imutil->matutil;
  self->findMatches = findMatchesImpl;
  return self;
}
