#include <matrix.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

int main(int argc, char **argv)
{
  int w = (argc > 1) ? atoi(argv[1]) : 3;
  int h = (argc > 2) ? atoi(argv[2]) : 3;

  Matrix *a, *c, *b;
  Shape shape = {w,h};
  new_empty_matrix(&a,shape);
  uniform(a,shape);
  print_matrix(a,"Uniform");

  return 0;
}
