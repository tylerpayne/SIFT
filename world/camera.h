typedef struct Camera Camera;

typedef Matrix* (*getCamPropFunc)(Camera*);
typedef Matrix* (*camPointFunc)(Camera*,Matrix*);
typedef void (*voidCamPointFunc)(Camera*,Matrix*);
typedef void (*setCamIntrinsicsFunc)(Camera*,float,float,int,int);

struct Camera
{
  MatrixUtil* matutil;
  Matrix* K;
  Matrix* Kinv;
  Matrix* R;
  Image* frame;

  setCamIntrinsicsFunc setIntrinsics;
  voidCamPointFunc setPosition;
  voidCamPointFunc setRotation;

  getCamPropFunc getPosition;
  getCamPropFunc getRotation;
  getCamPropFunc getForwardVector;

  camPointFunc projectImagePointToWorldPoint;
  camPointFunc projectWorldPointToImagePoint;
};

Camera* NewCamera(MatrixUtil* matutil, Matrix* K, Matrix* R, Matrix* frame);
