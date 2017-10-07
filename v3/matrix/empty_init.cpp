#include <matrix.h>

template<typename K>
Matrix<K>::Matrix(Tuple<int> s)
{
  empty_init(s,true);
}

template<typename K>
void Matrix<K>::empty_init(Tuple<int> s, bool isOnHost)
{
  basic_init(s,isOnHost);
  if (isOnHost)
  {
    size_t sz = sizeof(K)*shape.product();
    host_ptr = (K*)malloc(sz);
    memset((void *)host_ptr,0,sz);
    dev_ptr = NULL;
  } else
  {
    size_t sz = sizeof(K)*shape.product();
    cudaMalloc((void**)&dev_ptr,sz);
    cudaMemset((void*)dev_ptr,0,sz);
    host_ptr = NULL;
  }
}
