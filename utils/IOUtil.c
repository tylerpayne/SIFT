#include <utils/IOUtil.h>

#ifdef __cplusplus
  extern "C" {
#endif

Image* loadImageFromFileImpl(IOUtil* self, char* filepath)
{
  GError* err = NULL;
  const char* path = filepath;

  GdkPixbuf* pixbuf = gdk_pixbuf_new_from_file(path, &err);
  if (err != NULL)
  {
    printf("ERROR SAVING IMAGE!\n");
    printf("err code = %i",err[0]);
    return NULL;
  }
  int w = gdk_pixbuf_get_width(pixbuf);
  int h = gdk_pixbuf_get_height(pixbuf);
  int nChannels = gdk_pixbuf_get_n_channels(pixbuf);
  char* image = (char*)gdk_pixbuf_get_pixels(pixbuf);
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

  Shape shape = {w,h};
  Image* retval = self->imutil->newImage(self->imutil,data,shape);
  retval->nChannels = 1;//nChannels;
  retval->pixbuf = (void*)pixbuf;
  return retval;
}



void saveImageToFileImpl(IOUtil* self, Image* saveim, char* filename, IMTYPE filetype)
{
  GError* err = NULL;

  int fileLength = strlen(filename);
  char* ext = IMFORMATS[filetype];

  int extensionLength = strlen(ext) + 1;
  char* extension = (char*)malloc(sizeof(char)*extensionLength);
  char* dot = ".";
  extension[0] = dot[0];
  memcpy(&extension[1],ext,sizeof(char)*(extensionLength-1));

  char* path = (char*)malloc(sizeof(char)*(fileLength+extensionLength));
  memcpy(path,filename,sizeof(char)*fileLength);
  memcpy(&path[fileLength],extension,sizeof(char)*extensionLength);

  unsigned char* saveData = (unsigned char*)malloc(sizeof(unsigned char)*saveim->shape.width*saveim->shape.height*3);
  float max = ((self->imutil)->matutil)->maxVal(self->imutil->matutil,saveim->pixels);
  printf("MAX: %f",max);
  Image* normIm = self->imutil->multiplyC(self->imutil,saveim,(1.0/max)*255);
  normIm->syncHostFromDevice(normIm);
  int q = 0;
  for (int i = 0; i < normIm->shape.height; i++)
  {
    for (int j = 0; j < normIm->shape.width; j++)
    {
      Point2 point = {j,i};
      unsigned char v = (unsigned char)min(abs((normIm->pixels->getElement(normIm->pixels,point))),255);
      saveData[q] = v;
      saveData[q+1] = v;
      saveData[q+2] = v;
      q+=3;
    }
  }

  GdkPixbuf* savepixbuf = gdk_pixbuf_new_from_data(saveData,
                                                         GDK_COLORSPACE_RGB,
                                                         0,
                                                         8,
                                                         normIm->shape.height,
                                                         normIm->shape.width,
                                                         normIm->shape.height*sizeof(unsigned char)*3,
                                                         NULL,
                                                         NULL);

  gdk_pixbuf_save (savepixbuf, path, IMFORMATS[filetype], &err, NULL);

  if (err != NULL)
  {
    printf("ERROR SAVING IMAGE!\n");
    printf("err code = %i",err[0]);
  }
}

char* appendNumberToFilenameImpl(char* filename, int number)
{
  int length = strlen(filename);
  char* id;
  int offset;
  if (number < 10)
  {
    id = (char*)malloc(sizeof(char));
    id[0] = number + '0';
    offset = 1;
  } else if (number < 100)
  {
    id = (char*)malloc(sizeof(char)*2);
    id[0] = ((number/10)%10) + '0';
    id[1] = (number%10) + '0';
    offset = 2;
  } else if (number <1000)
  {
    id = (char*)malloc(sizeof(char)*3);
    id[0] = ((number/100)%100) + '0';
    id[1] = (number%100) + '0';
    id[2] = (number%10) + '0';
    offset =3;
  } else if (number < 10000)
  {
    id = (char*)malloc(sizeof(char)*3);
    id[0] = ((number/1000)%1000) + '0';
    id[1] = ((number/100)%100) + '0';
    id[2] = (number%100) + '0';
    id[3] = (number%10) + '0';
    offset = 4;
  }
  char* retval = (char*)malloc(sizeof(char)*(length+offset));
  memcpy(&retval[0],filename,sizeof(char)*length);
  memcpy(&retval[length],id,sizeof(char)*offset);
  return retval;
}

DLLEXPORT IOUtil* GetIOUtil(ImageUtil* imutil)
{
  IOUtil* self = (IOUtil*)malloc(sizeof(IOUtil));
  self->imutil = imutil;
  self->loadImageFromFile = loadImageFromFileImpl;
  self->saveImageToFile = saveImageToFileImpl;
  self->appendNumberToFilename = appendNumberToFilenameImpl;
  return self;
}

#ifdef __cplusplus
  }
#endif
