#include <gtk/gtk.h>

typedef struct IOUtil IOUtil;

typedef Image* (*loadImageFromFileFunc)(IOUtil*,char*);
typedef void (*saveImageToFileFunc)(IOUtil*, Image*, char*, int);


struct IOUtil
{
  ImageUtil* imutil;
  loadImageFromFileFunc loadImageFromFile;
  saveImageToFileFunc saveImageToFile;
};

IOUtil* GetIOUtil(ImageUtil* imutil);
