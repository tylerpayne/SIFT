#include <structs/Matrix.h>

#ifndef _IMAGE_
#define _IMAGE_

typedef struct Image Image;

typedef void (*imFunc)(Image*);

struct Image
{
  Matrix* pixels;
  Shape shape;
  int nChannels;

  void* pixbuf;

  imFunc free;
};
#endif
