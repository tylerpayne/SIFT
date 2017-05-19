#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;

int fill(Matrix *a, float b)
{
  memassert(a,DEVICE);

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  fill_kernel<<<gdim,bdim,0,_cudaStream>>>(a->dev_ptr,b,a->shape);
  return 0;
}

#ifdef __cplusplus
}
#endif
