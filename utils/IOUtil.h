#include <gtk/gtk.h>

typedef struct IOUtil IOUtil;

typedef Image* (*loadImageFromFileFunc)(IOUtil*,char*);
typedef void (*saveImageToFileFunc)(IOUtil*, Image*, char*, int);
typedef char* (*appendNumberToFilenameFunc)(char*,int);


struct IOUtil
{
  ImageUtil* imutil;
  loadImageFromFileFunc loadImageFromFile;
  saveImageToFileFunc saveImageToFile;
  appendNumberToFilenameFunc appendNumberToFilename;
};

IOUtil* GetIOUtil(ImageUtil* imutil);
