#ifndef GLIB_MAJOR_VERSION
  #include <gtk/gtk.h>
#endif

typedef struct DrawUtil DrawUtil;

typedef GtkWidget* (*newWindowFunc)(DrawUtil*);
typedef void (*drawImageFunc)(DrawUtil*,Image*,GtkWidget*);
typedef void (*drawKeypointsFunc)(DrawUtil*,Array*,Image*,GtkWidget*);

struct DrawUtil
{
  GdkPixbuf* keypoint;
  GtkWidget* defaultWindow;
  newWindowFunc newWindow;
  drawImageFunc drawImage;
  drawKeypointsFunc drawKeypoints;
};

DrawUtil* GetDrawUtil();
