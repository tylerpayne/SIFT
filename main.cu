#include <chai.h>


int main(int argc, char** argv)
{
  chai::cuda::init_cuda_libs();
  printf("1!\n");
  chai::Tuple<int> t({3,3});
  printf("2!\n");
  chai::Matrix<int> m(t);
  printf("3!\n");
  chai::Matrix<int> n = m({2},{1,3});
  printf("4!\n");
  chai::cuda::destroy_cuda_libs();
  return 0;
}
