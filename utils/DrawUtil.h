#include <string.h>
#include <gtk/gtk.h>
#include <utils/MatrixUtil.h>
#include <utils/ImageUtil.h>
#include <structs/Keypoint.h>

#ifndef _DRAWUTIL_
#define _DRAWUTIL_

typedef struct DrawUtil DrawUtil;

typedef GtkWidget* (*newWindowFunc)(DrawUtil*);
typedef void (*drawImageFunc)(DrawUtil*,Image*,GtkWidget*);
typedef void (*drawKeypointsFunc)(DrawUtil*,Array*,Image*,GtkWidget*);

struct DrawUtil
{
  GdkPixbuf* keypoint;
  newWindowFunc newWindow;
  drawImageFunc drawImage;
  drawKeypointsFunc drawKeypoints;
};

DLLEXPORT DrawUtil* GetDrawUtil();

#endif
