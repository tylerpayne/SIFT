#include <matrix.h>

template<typename K>
Matrix<K>::Matrix(K* ptr, bool isHostPtr, Tuple<int> s)
{
  basic_init(s,isHostPtr);
  if (isHostPtr)
  {
    host_ptr = ptr;
    dev_ptr = NULL;
  } else
  {
    dev_ptr = ptr;
    host_ptr = NULL;
  }
}
