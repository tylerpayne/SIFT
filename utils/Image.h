#include <nppi.h>

typedef struct Image Image;

struct Image
{
  int nChannels;
  Matrix* pixels;
  NppiSize shape;
};
