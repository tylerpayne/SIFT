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

General 1d object for indexing and other simple lists.

````C++
template <typename K>
class Tuple
{
public:
  K *components;
  int length;
  Tuple();
  Tuple(Tuple<K> *t);
  Tuple(std::initializer_list<K> coords);

  K operator()(int i);

  K prod();
  K norm();
  K norm(int l);
};
````

Matrix Initialization

````C++
Matrix(std::initializer_list<int> s);
Matrix(Tuple<int> &s);
Matrix(K* ptr, bool isHostPtr, Tuple<int> &s);
Matrix(K c, bool onHost, Tuple<int> &s);
````

Matrix Indexing & Copying

````C++
Matrix operator()(std::initializer_list<int> rows, std::initializer_list<int> cols);
Matrix operator()(Tuple<int> &rows, Tuple<int> &cols);
````

Arithmetic

````C++
Matrix operator+(Matrix<K> m);
````

Matrix Operations

````C++

````

Logic

````C++

````

Trig

````C++

````

Math etc

````C++

````

Statistics

````C++

````

Image

````C++

````

Signal
````C++

````

#### disclaimer
chai is an academic project of mine and is therefore unstable and constantly under development. I do not claim it to be production viable.
