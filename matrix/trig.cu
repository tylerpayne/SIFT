#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

int mcos(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  cos_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int msin(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  sin_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int mtan(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  tan_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int macos(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  acos_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int masin(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  asin_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int matan(Matrix *a, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  atan_kernel<<<gdim,bdim>>>(a->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int matan2(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  atan2_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int mhypot(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  hypot_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

#ifdef __cplusplus
}
#endif
