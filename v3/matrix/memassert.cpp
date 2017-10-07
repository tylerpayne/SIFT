#include <matrix.h>

extern cudaStream_t _cudaStream;
template<typename K>
int Matrix<K>::memassert(Matrix m, int dest)
{
  if (m==NULL) printf("null assert error\n");
  if (m.isHostSide != dest)
  {
    size_t size = sizeof(K)*shape.product();
    if (dest == DEVICE)
    {
      K *dev_ptr;
      cudaMalloc((void**)&dev_ptr,size);
      cudaMemcpyAsync(dev_ptr,m.host_ptr,size,cudaMemcpyHostToDevice,_cudaStream);
      m.dev_ptr = dev_ptr;
      m.isHostSide = false;
    }
    else if (dest == HOST)
    {
      cudaMemcpyAsync(m.host_ptr,m.dev_ptr,size,cudaMemcpyDeviceToHost,_cudaStream);
      cudaFree(m.dev_ptr);
      m.dev_ptr = NULL;
      m.isHostSide = true;
    }
  }
  return 0;
}
