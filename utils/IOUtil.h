#include "ImageUtil.h"

typedef struct IOUtil IOUtil;

typedef Image* (*loadImageFromFileFunc)(char*);
typedef void (*saveImageToFileFunc)(Image*,char*);


struct IOUtil
{
  ImageUtil* imutil;
  loadImageFromFileFunc loadImageFromFile;
  saveImageToFileFunc saveImageToFile;
};
