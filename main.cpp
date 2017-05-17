#include <matrix.h>

int main(int argc, char **argv)
{
  init_cuda_libs();
  int w = (argc > 1) ? atoi(argv[1]) : 3;
  int h = (argc > 2) ? atoi(argv[2]) : 3;

  Matrix *l_large, *l, *diff, *k1, *k2, *o;
  load_image(&l_large,"left.png");

  int divisor = 4;
  Shape small = {l_large->shape.width/divisor,l_large->shape.height/divisor};
  resample(l_large,&l,small);

  Shape kernel_shape = {15,15};
  generate_gaussian(&k1,kernel_shape,1,1);
  generate_gaussian(&k2,kernel_shape,3,3);

  new_empty_matrix(&diff,kernel_shape);
  subtract(k1,k2,diff);

  new_empty_matrix(&o,small);
  convolve(l,diff,o);

  write_image(o,"left_test.png");
  destroy_cuda_libs();
  return 0;
}
