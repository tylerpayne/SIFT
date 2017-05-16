#include <matrix.h>

int memassert(Matrix *m, int dest)
{
  if (m==NULL) return -111;
  if (m->isHostSide != dest)
  {
    size_t size = sizeof(float)*SHAPE2LEN(m->shape);
    if (dest == DEVICE)
    {
      float *dev_ptr;
      cuda_safe_call(cudaMalloc((void**)&dev_ptr,size));
      cuda_safe_call(cudaMemcpy(dev_ptr,m->host_ptr,size,cudaMemcpyHostToDevice));
      m->dev_ptr = dev_ptr;
      m->isHostSide = FALSE;
    }
    else if (dest == HOST)
    {
      cuda_safe_call(cudaMemcpy(m->host_ptr,m->dev_ptr,size,cudaMemcpyDeviceToHost));
      cuda_safe_call(cudaFree(m->dev_ptr));
      m->dev_ptr = NULL;
      m->isHostSide = TRUE;
    }
  }
  return 0;
}
