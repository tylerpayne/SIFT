#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

int msqrt(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  sqrt_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int mabs(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  abs_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}


int mexp(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  exp_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}


int mlog(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  log_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}


int mpow(Matrix *a, float e, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  pow_kernel<<<gdim,bdim>>>(a->dev_ptr,e,ret->dev_ptr,a->shape);
  return 0;
}

#ifdef __cplusplus
}
#endif
