#include <chai.h>

namespace chai {
  namespace cuda {

    extern __device__ int tuple_product(int_tuple shape);
    extern cublasHandle_t _cublasHandle;

    ///////////////
    // K = FLOAT //
    ///////////////

    template<>
    matrix<float> dot_kernel(matrix<float> &a, matrix<float> &b)
    {
      assert(a.shape(1) == b.shape(0));
      matrix<float>::memassert(a, DEVICE);
      matrix<float>::memassert(b, DEVICE);
      const tuple<int> outshape({a.shape(1),b.shape(0)});
      matrix<float> out(outshape);
      matrix<float>::memassert(out, DEVICE);

      float alpha = 1;
      float beta = 0;
      int lda, tda, tdb;
      cublasOperation_t opA, opB;

      if (a.T)
      {
        opA = CUBLAS_OP_T;
        lda = a.shape(1);
        tda = a.shape(0);
      } else
      {
        opA = CUBLAS_OP_N;
        lda = a.shape(0);
        tda = a.shape(1);
      }

      if (b.T)
      {
        opB = CUBLAS_OP_T;
        tdb = b.shape(0);
      } else
      {
        opB = CUBLAS_OP_N;
        tdb = b.shape(1);
      }

      cublas_safe_call(
        cublasSgemm(_cublasHandle,
                               opA, opB,
                               lda, tdb, tda,
                               &alpha,
                               a.dev_ptr, a.shape(0),
                               b.dev_ptr, b.shape(0),
                               &beta,
                               out.dev_ptr, out.shape(0)
        )
      );
      return out;
    }

  }
}
