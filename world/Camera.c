#include "Camera.h"

void setIntrinsicPropertiesImpl(Camera* self, float mmFocalLength, float mmSensorWidth, int nPixelsWidth, int nPixelsHeight)
{
  float Ox = ((float)nPixelsWidth)/2.0;
  float Oy = ((float)nPixelsHeight)/2.0;
  float f = mmFocalLength*(((float)nPixelsWidth)/mmSensorWidth);
  float* data = (float*)malloc(sizeof(float)*9);
  data[0] = f;
  data[2] = Ox;
  data[4] = f;
  data[5] = Oy;
  data[8] = 1;
  Matrix* newK = self->matutil->newMatrix(data,3,3);
  Rect size = {3,3};
  Point2 Aidx = {0,0};
  Point2 Bidx = {0,0};
  self->matutil->copy(self->matutil,newK,self->K,size,Aidx,Bidx);
  self->matutil->inv(self->matutil,self->K,self->Kinv);
  newK->free(newK);
}

void setPositionCameraImpl(Camera* self, Matrix* xyz)
{
  Rect size = {1,3};
  Point2 Aidx = {0,0};
  Point2 Bidx = {3,0};
  self->matutil->copy(self->matutil,xyz,self->R,size,Aidx,Bidx);
}

void setRotationCameraImpl(Camera* self, Matrix* xyz)
{
  Matrix* X = self->matutil->newEmptyMatrix(3,3);
  Matrix* Y = self->matutil->newEmptyMatrix(3,3);
  Matrix* Z = self->matutil->newEmptyMatrix(3,3);

  float sinx = sinf(xyz->getElement(xyz,0,0));
  float cosx = cosf(xyz->getElement(xyz,0,0));
  float siny = sinf(xyz->getElement(xyz,1,0));
  float cosy = cosf(xyz->getElement(xyz,1,0));
  float sinz = sinf(xyz->getElement(xyz,2,0));
  float cosz = cosf(xyz->getElement(xyz,2,0));
  X->setElement(X,0,0, 1.0);
  X->setElement(X,1,1,cosx);
  X->setElement(X,1,2,-sinx);
  X->setElement(X,2,1,sinx);
  X->setElement(X,2,2,cosx);

  Y->setElement(Y,1,1, 1.0);
  Y->setElement(Y,0,0,cosy);
  Y->setElement(Y,0,2,siny);
  Y->setElement(Y,2,0,-siny);
  Y->setElement(Y,2,2,cosy);

  Z->setElement(Z,2,2, 1.0);
  Z->setElement(Z,0,0,cosz);
  Z->setElement(Z,0,1,-sinz);
  Z->setElement(Z,1,0,sinz);
  Z->setElement(Z,1,1,cosz);
  Matrix* YX = self->matutil->newEmptyMatrix(3,3);
  Matrix* R = self->matutil->newEmptyMatrix(3,3);
  self->matutil->dot(self->matutil,Y,X,YX);
  self->matutil->dot(self->matutil,Z,YX,R);

  Rect size = {3,3};
  Point2 Aidx = {0,0};
  Point2 Bidx = {0,0};
  self->matutil->copy(self->matutil,R,self->R,size,Aidx,Bidx);
}

Matrix* getPositionCameraImpl(Camera* self)
{
  Matrix* retval = self->matutil->newEmptyMatrix(3,1);
  Rect size = {1,3};
  Point2 Aidx = {3,0};
  Point2 Bidx = {0,0};
  self->matutil->copy(self->matutil,self->R,retval,size,Aidx,Bidx);
  return retval;
}

Matrix* getRotationMatrixCameraImpl(Camera* self)
{
  Matrix* retval = self->matutil->newEmptyMatrix(3,3);
  Rect size = {3,3};
  Point2 Aidx = {0,0};
  Point2 Bidx = {0,0};
  self->matutil->copy(self->matutil,self->R,retval,size,Aidx,Bidx);
  return retval;
}

/*Matrix* getForwardVectorCameraImpl(Camera* self)
{

}*/

Matrix* projectImagePointToWorldPointCameraImpl(Camera* self, Matrix* point)
{
  Matrix* pc = self->matutil->newEmptyMatrix(3,1);
  Matrix* pw = self->matutil->newEmptyMatrix(4,1);

  self->matutil->dot(self->matutil,self->Kinv,point,pc);
  self->R->T = CUBLAS_OP_T;
  self->matutil->dot(self->matutil,self->R,pc,pw);
  self->R->T = CUBLAS_OP_N;
  return pw;
}

Matrix* projectWorldPointToImagePointCameraImpl(Camera* self, Matrix* point)
{
  Matrix* pc = self->matutil->newEmptyMatrix(3,1);
  Matrix* pi = self->matutil->newEmptyMatrix(3,1);

  self->matutil->dot(self->matutil,self->R,point,pc);
  self->matutil->dot(self->matutil,self->K,pc,pi);
  return pi;
}

Camera* NewCamera(MatrixUtil* matutil, Matrix* K, Matrix* R, Image* frame)
{
  Camera* self = (Camera*)malloc(sizeof(Camera));
  if (K != NULL)
  {
    self->K = K;
    self->Kinv = matutil->newEmptyMatrix(3,3);
    matutil->inv(matutil,K,self->Kinv);
  } else
  {
    self->K = matutil->newEmptyMatrix(3,3);
    self->Kinv = matutil->newEmptyMatrix(3,3);
  }

  if (R != NULL)
  {
    self->R = R;
  } else
  {
    self->R = matutil->newEmptyMatrix(3,4);
  }

  if (frame != NULL)
  {
    self->frame = frame;
  }
  self->matutil = matutil;
  self->setIntrinsics = setIntrinsicPropertiesImpl;
  self->setPosition = setPositionCameraImpl;
  self->setRotation = setRotationCameraImpl;
  self->getPosition = getPositionCameraImpl;
  self->getRotation = getRotationMatrixCameraImpl;
  self->projectWorldPointToImagePoint = projectWorldPointToImagePointCameraImpl;
  self->projectImagePointToWorldPoint = projectImagePointToWorldPointCameraImpl;

  return self;
}
