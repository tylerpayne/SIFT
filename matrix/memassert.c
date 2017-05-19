#include <matrix.h>

extern cudaStream_t _cudaStream;

int memassert(Matrix *m, int dest)
{
  if (m==NULL) printf("null assert error\n");
  if (m->isHostSide != dest)
  {
    size_t size = sizeof(float)*SHAPE2LEN(m->shape);
    if (dest == DEVICE)
    {
      float *dev_ptr;
      cuda_safe_call(cudaMalloc((void**)&dev_ptr,size));
      cuda_safe_call(cudaMemcpyAsync(dev_ptr,m->host_ptr,size,cudaMemcpyHostToDevice,_cudaStream));
      m->dev_ptr = dev_ptr;
      m->isHostSide = FALSE;
    }
    else if (dest == HOST)
    {
      cuda_safe_call(cudaMemcpyAsync(m->host_ptr,m->dev_ptr,size,cudaMemcpyDeviceToHost,_cudaStream));
      cuda_safe_call(cudaFree(m->dev_ptr));
      m->dev_ptr = NULL;
      m->isHostSide = TRUE;
    }
  }
  return 0;
}
