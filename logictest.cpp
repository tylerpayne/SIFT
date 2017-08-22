#include <matrix.h>

int main(int argc, char const *argv[]) {
  init_cuda_libs();

  Matrix *a, *b;
  Shape shape = {9,9};

  new_empty_matrix(&a,shape);

  linfill(a,1.0,10.0);

  print_matrix(a,"a");

  free_matrix(a);

  destroy_cuda_libs();
  return 0;
}
