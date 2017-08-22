#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;

int copy_region(Matrix *a, Matrix *out, Point2 a_idx, Point2 out_idx, Shape shape)
{
  memassert(a, DEVICE);
  memassert(out, DEVICE);

  dim3 bdim,gdim;
  make_launch_parameters(shape,2,&bdim,&gdim);
  copy_region_kernel<<<gdim,bdim,0,_cudaStream>>>(a->dev_ptr,out->dev_ptr,a_idx,out_idx,shape,a->shape,out->shape);

  return 0;
}

#ifdef __cplusplus
}
#endif
