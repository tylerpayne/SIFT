#include <string.h>
#include <gtk/gtk.h>
#include <utils/MatrixUtil.h>
#include <utils/ImageUtil.h>

#ifndef _IOUTIL_
#define _IOUTIL_

typedef struct IOUtil IOUtil;

typedef Image* (*loadImageFromFileFunc)(IOUtil*,char*);
typedef void (*saveImageToFileFunc)(IOUtil*, Image*, char*, IMTYPE);
typedef char* (*appendNumberToFilenameFunc)(char*,int);


struct IOUtil
{
  ImageUtil* imutil;
  loadImageFromFileFunc loadImageFromFile;
  saveImageToFileFunc saveImageToFile;
  appendNumberToFilenameFunc appendNumberToFilename;
};

DLLEXPORT IOUtil* GetIOUtil(ImageUtil* imutil);

#endif
