#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

int print_matrix(Matrix *a, const char *msg)
{
  memassert(a,HOST);
  printf("MATRIX: %s\n",msg);
  printf("SHAPE: (h,w): (%i,%i)\n\n",a->shape.height,a->shape.width);
  for (int i = 0; i < a->shape.height; i ++)
  {
    for (int j = 0; j < a->shape.width; j++)
    {
      Point2 id = {j,i};
      printf("[%3.3f]",get_element(a,id));
    }
    printf("\n");
  }
  printf("\n\n");
}

#ifdef __cplusplus
}
#endif
