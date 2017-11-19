# chai
chai is a no-frills CUDA accelerated matrix library written in C++11.

# usage
a basic example:
````C++
#include <chai.h>
using namespace chai;
int main(int argc, char **argv)
{
  cuda::init_cuda_libs(); // must call before using chai

  Tuple<int> s1({10,20});

  Matrix<int> m1(s1);
  Matrix<int> m2({10,20});

  Matrix<int> m3 = m1 + m2;

  cuda::destroy_cuda_libs(); // best practice
  return 0;
}
````

# reference

#### architecture

````C++
namespace chai
{
  class Tuple;
  class Matrix;
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
class Tuple
{
public:
  //// VARIABLES ////
  K *components;
  int length;

  //// CONSTRUCTORS ////
  Tuple();
  Tuple(Tuple<K> *t);
  Tuple(std::initializer_list<K> coords);

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
class Matrix
{
public:
  //// MEMBER VARIABLES ////
  bool isHostSide, T;
  K *host_ptr, *dev_ptr;
  Tuple<int> shape;
  cudaStream_t stream;

  //// STATIC FUNCTIONS ////
  static void basic_init(Matrix<K> *m, Tuple<int> &s, bool isOnHost);
  static void empty_init(Matrix<K> *m, Tuple<int> &s, bool isOnHost);
  static void const_init(Matrix<K> *m, K val, Tuple<int> &s, bool isOnHost);

  static void memassert(Matrix<K> *m, int dest);

  //// CONSTRUCTORS ////
  Matrix(std::initializer_list<int> s);
  Matrix(Tuple<int> &s);
  Matrix(K* ptr, bool isHostPtr, Tuple<int> &s);
  Matrix(K c, bool onHost, Tuple<int> &s);

  //// OPERATORS ////
  Matrix operator()(std::initializer_list<int> rows, std::initializer_list<int> cols);
  Matrix operator()(Tuple<int> &rows, Tuple<int> &cols);
  Matrix operator+(Matrix<K> m);
};
````

#### disclaimer
chai is an academic project of mine and is therefore unstable and constantly under development. I do not claim it to be production viable.
