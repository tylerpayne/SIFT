#include <chai.h>

using namespace chai;

int main(int argc, char** argv)
{
  cuda::init_cuda_libs();


  tuple<int> t1({1,2,3,4,5});
  tuple<int> t2({1,2,3,4,5});

  int *ones = (int*)malloc(sizeof(int)*9);
  memset(ones,0,sizeof(int)*9);
  matrix<int> m1(ones,true,{3,3});
  matrix<int> m2(ones,true,{3,3});

  matrix<int> m3 = m1+m2;

  m3.print();

  cuda::destroy_cuda_libs();
  return 0;
}
