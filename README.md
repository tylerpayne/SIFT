# chai
chai is a C++ matrix library accelerated by CUDA.

# usage
a basic example:
````C++
#include <chai.h>
using namespace chai;
int main(int argc, char **argv)
{
  cuda::init_cuda_libs(); // must call before using chai

  tuple<int> s1({10,20});

  matrix<int> m1(s1);
  matrix<int> m2({10,20});

  matrix<int> m3 = m1 + m2;

  cuda::destroy_cuda_libs(); // best practice
  return 0;
}
````

# reference

#### architecture

````C++
namespace chai
{
  class tuple;
  class matrix;
  namespace cuda
  {
    void init_cuda_libs();
    void destroy_cuda_libs();
    ...
  }
}
````

#### the tuple class

````C++
template <typename K>
class tuple
{
public:
  //// VARIABLES ////
  K *components;
  int length;

  //// CONSTRUCTORS ////
  tuple();
  tuple(tuple<K> *t);
  tuple(std::initializer_list<K> coords);

  //// OPERATORS ////
  K operator()(int i);

  //// FUNCTIONS ////
  K prod();
  K norm();
  K norm(int l);
};
````
#### the matrix class

````C++
template <typename K>
class matrix
{
public:
  //// MEMBER VARIABLES ////
  bool isHostSide, T;
  K *host_ptr, *dev_ptr;
  tuple<int> shape;
  cudaStream_t stream;

  //// STATIC FUNCTIONS ////
  static void basic_init(matrix<K> *m, tuple<int> &s, bool isOnHost);
  static void empty_init(matrix<K> *m, tuple<int> &s, bool isOnHost);
  static void const_init(matrix<K> *m, K val, tuple<int> &s, bool isOnHost);

  static void memassert(matrix<K> *m, int dest);

  //// CONSTRUCTORS ////
  matrix(std::initializer_list<int> s);
  matrix(tuple<int> &s);
  matrix(K* ptr, bool isHostPtr, tuple<int> &s);
  matrix(K c, bool onHost, tuple<int> &s);

  //// OPERATORS ////
  matrix operator()(std::initializer_list<int> rows, std::initializer_list<int> cols);
  matrix operator()(tuple<int> &rows, tuple<int> &cols);
  matrix operator+(matrix<K> m);
};
````

#### disclaimer
chai is an academic project of mine and is therefore unstable and constantly under development. I do not claim it to be production viable.
