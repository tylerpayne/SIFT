typedef struct Image Image;
typedef void (*freeImageFunc)(Image*);
struct Image
{
  int nChannels;
  Matrix* pixels;
  int* shape;
  freeImageFunc free;
  void* pixbuf;
};
