#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <core.h>

template <typename K>
class Matrix
{
private:
  void basic_init(Tuple<int> s, bool isOnHost);
  void empty_init(Tuple<int> s, bool isOnHost);
  void const_init(K val, Tuple<int> s, bool isOnHost);

  //############//
  // UTIL //
  //##############//
  int memassert(Matrix m, int dest);

public:
  //##########//
  // PARAMS //
  //#########//
  bool isHostSide, T;
  K *host_ptr, *dev_ptr;
  Tuple<int> shape;

  //#################//
  // CONSTRUCTORS //
  //##############//
  Matrix(Tuple<int> s);
  Matrix(K* ptr, bool isHostPtr, Tuple<int> s);
  Matrix(K c, bool isOnHost, Tuple<int> s);

  //############//
  // DESTRUCTOR //
  //############//
  ~Matrix()
  {
    if (isHostSide)
    {
      free(host_ptr);
    } else
    {
      cuda_safe_call(cudaFree(dev_ptr));
    }
    host_ptr = NULL;
    dev_ptr = NULL;
  }
};




//#include <matrix_funcs.h>
#endif
