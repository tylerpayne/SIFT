#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

int add(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  add_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int divide(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  divide_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int multiply(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  multiply_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int subtract(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  subtract_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

#ifdef __cplusplus
}
#endif
