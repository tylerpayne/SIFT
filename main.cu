#include <chai.h>

using namespace chai;

int main(int argc, char **argv) {
  cuda::init_cuda_libs();

  int a = 5;
  int b = 2;

  matrix<int> m1(a,true,{3,3});
  //matrix<int> m2(b,true,{3,3});

  m1 += b;

  m1.print();

  cuda::destroy_cuda_libs();
  return 0;
}
