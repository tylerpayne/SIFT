#include <matrix.h>

template<typename K>
Matrix<K>::Matrix(K val, bool isOnHost, Tuple<int> s)
{
  empty_init(s,isOnHost);
  if (isOnHost)
  {
    K* tmp_ptr = host_ptr;
    for (int i = 0; i < shape.product(); i++)
    {
      *tmp_ptr = val;
      tmp_ptr++;
    }
  } else
  {
    printf("to_do\n");
    //CALL CUDA FILL KERNEL
  }
}
