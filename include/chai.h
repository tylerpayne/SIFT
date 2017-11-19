#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 1024
#endif

#define HOST 1
#define DEVICE 0

#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <cublas_v2.h>
#include <curand.h>
#include <npp.h>
#include <nppi.h>
//#include <lodepng.h>
//#include <errors.h>
#include <assert.h>
#include <complex.h>

namespace chai
{
  template <typename K>
  class Tuple
  {
  public:
    K *components;
    int length;
    Tuple();
    Tuple(Tuple<K> *t);
    Tuple(std::initializer_list<K> coords);

    K operator()(int i);

    K prod();
    K norm();
    K norm(int l);
  };

  template<typename K>
  class Rect
  {
  public:
    Tuple<K> origin;
    Tuple<int> shape;
    Rect(Tuple<K> p, Tuple<int> s) {origin = p; shape = s;}
  };

  template <typename K>
  class Matrix
  {
  public:
    bool isHostSide, T;
    K *host_ptr, *dev_ptr;
    Tuple<int> shape;
    cudaStream_t stream;

    static void basic_init(Matrix<K> *m, Tuple<int> &s, bool isOnHost);
    static void empty_init(Matrix<K> *m, Tuple<int> &s, bool isOnHost);
    static void const_init(Matrix<K> *m, K val, Tuple<int> &s, bool isOnHost);

    static void memassert(Matrix<K> *m, int dest);

    Matrix(std::initializer_list<int> s);
    Matrix(Tuple<int> &s);
    Matrix(K* ptr, bool isHostPtr, Tuple<int> &s);
    Matrix(K c, bool onHost, Tuple<int> &s);

    Matrix operator()(std::initializer_list<int> rows, std::initializer_list<int> cols);
    Matrix operator()(Tuple<int> &rows, Tuple<int> &cols);
    Matrix operator+(Matrix<K> m);

    ~Matrix();
  };

  namespace cuda
  {
    void init_cuda_libs();
    void destroy_cuda_libs();
    void set_stream(cudaStream_t s);

    template <typename K>
    void safe_call(K err);

    void cublas_safe_call(cublasStatus_t err);
    void cuda_safe_call(cudaError_t err);
    void npp_safe_call(NppStatus err);
    void curand_safe_call(curandStatus_t err);

    __device__ int tuple_product(Tuple<int> shape);
    __device__ void C2IDX_kernel(int i, int shape[2], int *r, int *c);
    __device__ int IDX2C_kernel(int index[2], int shape[2]);
  }

  int IDX2C(Tuple<int> index, Tuple<int> shape);
  Tuple<int> C2IDX(int i, Tuple<int> shape);
  void make_launch_parameters(Tuple<int> shape, int dim, dim3 *bdim, dim3 *gdim);
}

#include <kernels.h>
#include <tuple.h>
#include <matrix.h>
