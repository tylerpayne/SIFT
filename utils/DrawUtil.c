#include "DrawUtil.h"


GtkWidget* newWindowImpl(DrawUtil* self)
{
  GtkWidget* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  if (self->defaultWindow == NULL)
  {
    self->defaultWindow = window;
  }
  return window;
}

void drawImageImpl(DrawUtil* self, Image* im, GtkWidget* window)
{
  if (window == NULL)
  {
    if (self->defaultWindow == NULL)
    {
      self->newWindow(self);
    }
    GtkWidget* imageWidget = gtk_image_new_from_pixbuf((GdkPixbuf*)(im->pixbuf));
    gtk_container_add(GTK_CONTAINER(self->defaultWindow),imageWidget);
    gtk_widget_show_all(self->defaultWindow);
  }
  else
  {
    GtkWidget* imageWidget = gtk_image_new_from_pixbuf((GdkPixbuf*)(im->pixbuf));
    gtk_container_add(GTK_CONTAINER(window),imageWidget);
    gtk_widget_show_all(window);
  }
}

void drawKeypointsImpl(DrawUtil* self, Array* keypoints, Image* im, GtkWidget* window)
{
  if (window == NULL)
  {
    if (self->defaultWindow == NULL)
    {
      self->newWindow(self);
    }
    GdkPixbuf* displaybuf = gdk_pixbuf_copy((GdkPixbuf*)(im->pixbuf));
    Keypoint** kpList = (Keypoint**)(keypoints->ptr);
    int radius = 9;
    for (int i = 0; i < keypoints->count; i++)
    {
      GdkPixbuf* src = gdk_pixbuf_copy(self->keypoint);
      Keypoint* kp = kpList[i];
      int dest_x = (int)(kp->position[0]) - radius;
      int dest_y = (int)(kp->position[1]) - radius;
      if (dest_x >= 0 && dest_y >= 0 && dest_x + radius*2 < im->shape[0] && dest_y + radius*2 < im->shape[1])
      {
        gdk_pixbuf_composite                (src,
                                                           displaybuf,
                                                           dest_x,
                                                           dest_y,
                                                           radius*2,
                                                           radius*2,
                                                           dest_x,
                                                           dest_y,
                                                           ((double)radius)/50.0,
                                                           ((double)radius)/50.0,
                                                           GDK_INTERP_HYPER,
                                                           255);
      }
    }
    GtkWidget* imageWidget = gtk_image_new_from_pixbuf(displaybuf);
    gtk_container_add(GTK_CONTAINER(self->defaultWindow),imageWidget);
    g_signal_connect(G_OBJECT(self->defaultWindow), "destroy",
      G_CALLBACK(gtk_main_quit), NULL);
    gtk_widget_show_all(self->defaultWindow);
  }
  else
  {
    GtkWidget* imageWidget = gtk_image_new_from_pixbuf((GdkPixbuf*)(im->pixbuf));
    gtk_container_add(GTK_CONTAINER(window),imageWidget);
    gtk_widget_show_all(window);
  }
}


DrawUtil* GetDrawUtil()
{
  DrawUtil* self = (DrawUtil*)malloc(sizeof(DrawUtil));
  self->defaultWindow = NULL;
  GError* err = NULL;
  const char* path = "im_keypoint.png";
  self->keypoint = gdk_pixbuf_new_from_file(path, &err);
  self->newWindow = newWindowImpl;
  self->drawImage = drawImageImpl;
  self->drawKeypoints = drawKeypointsImpl;
  return self;
}
