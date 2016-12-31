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
