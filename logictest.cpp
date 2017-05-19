#include <matrix.h>

int main(int argc, char const *argv[]) {
  init_cuda_libs();

  Matrix *a, *b;
  Shape shape = {9,9};

  new_empty_matrix(&a,shape);
  new_empty_matrix(&b,shape);

  fill(a,1.0);
  fill(b,0.9);

  gt(a,b,NULL);

  print_matrix(a,"a");

  free_matrix(a);
  free_matrix(b);


  destroy_cuda_libs();
  return 0;
}
