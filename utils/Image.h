#include <stdlib.h>
#include <nppi.h>

int VERBOSITY = 0;

static int IDX2C(int i, int j, int td)
{
  return (i*td)+j;
}


typedef struct Image Image;

struct Image
{
  int nChannels;
  Npp32f* pixels;
  NppiSize shape;
};
/*
typedef struct ScaleSpaceImage ScaleSpaceImage;

typedef Image* (*getImageAtFunc)(ScaleSpaceImage* self, int,int);
typedef void (*setImageAtScaleFunc)(ScaleSpaceImage* self, Image* img, int, int);

struct ScaleSpaceImage
{
  int nOctaves;
  int nScalesPerOctave;
  Image* scalespace;
  getImageAtFunc getImageAt;
  setImageAtScaleFunc setImageAt;
};
*/
