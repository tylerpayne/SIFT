#include <structs/Matrix.h>

#ifndef _IMAGE_
#define _IMAGE_

typedef struct Image Image;

  typedef void (*syncImageFunc)(Image*);
  typedef void (*freeImageFunc)(Image*);

struct Image
{
  Matrix* pixels;
  int nChannels;
  Shape shape;
  void* pixbuf;
  syncImageFunc syncDeviceFromHost;
  syncImageFunc syncHostFromDevice;
  freeImageFunc free;
};
#endif
