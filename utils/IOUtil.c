#include "IOUtil.h"
#include <gdk-pixbuf/gdk-pixbuf.h>

Image* loadImageFromFileImpl(IOUtil* self, char* filepath)
{
  Gerror* err;
  GdkPixbuf* pixbuf = (GdkPixbuf*)malloc(sizeof(pixbuf));
  const char* path = filepath;

  pixbuf = gdk_pixbuf_new_from_file(path, &err);
  if (err != NULL)
  {
    printf("ERROR SAVING IMAGE!\n");
    printf("err code = %i",err[0]);
    return NULL;
  }
  int w = gdk_pixbuf_get_width(pixbuf);
  int h = gdk_pixbuf_get_width(pixbuf);
  int nChannels = gdk_pixbuf_get_n_channels(pixbuf);
  int size = sizeof(float)*w*h;
  float* data = (float*)malloc(size);
  int counter = 0;
  for (int i = 0; i < w*h*nChannels; i+=nChannels)
  {
    if (nChannels >= 1)
    {
      data[counter] = 0.21*((float)(int)image[i])/255.0;
    }
    if (nChannels >= 2)
    {
      data[counter] += 0.72*((float)(int)image[i+1])/255.0;
    }
    if (nChannels >= 3)
    {
      data[counter] += 0.07*((float)(int)image[i+2])/255.0;
    }
    counter++;
  }

  Image* retval = self->imutil->newImage(self->imutil,data,w,h);
  retval->nChannels = nChannels;
  retval->pixbuf = (void*)pixbuf;
  return retval;
}



void saveImageFromFileImpl(IOUtil* self, Image* saveim, char* filepath, int imType)
{
  Gerror* err;
  const char* path = filepath;

  gdk_pixbuf_save ((GdkPixbuf*)(saveim->pixbuf), path, IMFORMATS[imType], &error, NULL);

  if (err != NULL)
  {
    printf("ERROR SAVING IMAGE!\n");
    printf("err code = %i",err[0]);
  }

}
