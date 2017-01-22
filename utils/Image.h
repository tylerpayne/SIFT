typedef struct Image Image;

  typedef void (*syncImageFunc)(Image*);
  typedef void (*freeImageFunc)(Image*);

struct Image
{
  Matrix* pixels;
  int nChannels;
  int* shape;
  void* pixbuf;
  syncImageFunc syncDeviceFromHost;
  syncImageFunc syncHostFromDevice;
  freeImageFunc free;
};
