#include <stdlib.h>

typedef struct Image Image;

struct Image
{
  int nChannels;
  Matrix* pixels;
  int* shape;
};

typedef struct ScaleSpaceImage ScaleSpaceImage;

typedef Image* (*getImageAtFunc)(ScaleSpaceImage* self, int,int);

struct ScaleSpaceImage
{
  int nOctaves;
  int nScalesPerOctave;
  Image* scalespace;
  getImageAtFunc getImageAt;
};
