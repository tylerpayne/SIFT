#include <matrix.h>

extern cublasHandle_t _cublasHandle;

void mdot(Matrix *a, Matrix *b, Matrix *out)
{
  memassert(a, DEVICE);
  memassert(b, DEVICE);
  memassert(out, DEVICE);

  float alpha = 1;
  float beta = 0;
  int lda, tda, tdb;
  cublasOperation_t opA, opB;

  if (a->T)
  {
    opA = CUBLAS_OP_T;
    lda = a->shape.width;
    tda = a->shape.height;
  } else
  {
    opA = CUBLAS_OP_N;
    lda = a->shape.height;
    tda = a->shape.width;
  }

  if (b->T)
  {
    opB = CUBLAS_OP_T;
    tdb = b->shape.height;
  } else
  {
    opB = CUBLAS_OP_N;
    tdb = b->shape.width;
  }

  cublas_safe_call(
    cublasSgemm(_cublasHandle,
                           opA, opB,
                           lda, tdb, tda,
                           &alpha,
                           a->dev_ptr, a->shape.height,
                           b->dev_ptr, b->shape.height,
                           &beta,
                           out->dev_ptr, out->shape.height
    )
  );
}
