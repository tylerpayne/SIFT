#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

int eq(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  eq_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int gt(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  gt_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int gte(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  gte_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int lt(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  lt_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

int lte(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a,DEVICE);
  memassert(b,DEVICE);
  memassert(out,DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  dim3 bdim,gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  lte_kernel<<<gdim,bdim>>>(a->dev_ptr,b->dev_ptr,ret->dev_ptr,a->shape);
  return 0;
}

#ifdef __cplusplus
}
#endif
