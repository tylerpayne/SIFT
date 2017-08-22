#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;

int linfill(Matrix *a, float from, float to)
{
  memassert(a,DEVICE);

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  float step = (to - from) / (float)SHAPE2LEN(a->shape);
  linfill_kernel<<<gdim,bdim,0,_cudaStream>>>(a->dev_ptr,from,step,a->shape);
  return 0;
}

#ifdef __cplusplus
}
#endif
