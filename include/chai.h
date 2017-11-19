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
  class tuple
  {
  public:
    K *components;
    int length;
    tuple();
    tuple(tuple<K> *t);
    tuple(std::initializer_list<K> coords);

    K operator()(int i);

    K prod();
    K norm();
    K norm(int l);
  };

  template<typename K>
  class Rect
  {
  public:
    tuple<K> origin;
    tuple<int> shape;
    Rect(tuple<K> p, tuple<int> s) {origin = p; shape = s;}
  };

  template <typename K>
  class matrix
  {
  public:
    bool isHostSide, T;
    K *host_ptr, *dev_ptr;
    tuple<int> shape;
    cudaStream_t stream;

    static void basic_init(matrix<K> *m, tuple<int> &s, bool isOnHost);
    static void empty_init(matrix<K> *m, tuple<int> &s, bool isOnHost);
    static void const_init(matrix<K> *m, K val, tuple<int> &s, bool isOnHost);

    static void memassert(matrix<K> *m, int dest);

    matrix(std::initializer_list<int> s);
    matrix(tuple<int> &s);
    matrix(K* ptr, bool isHostPtr, tuple<int> &s);
    matrix(K c, bool onHost, tuple<int> &s);

    matrix operator()(std::initializer_list<int> rows, std::initializer_list<int> cols);
    matrix operator()(tuple<int> &rows, tuple<int> &cols);
    matrix operator+(matrix<K> m);

    ~matrix();
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

    __device__ int tuple_product(tuple<int> shape);
    __device__ void c2idx_kernel(int i, int shape[2], int *r, int *c);
    __device__ int idx2c_kernel(int index[2], int shape[2]);
  }

  int idx2c(tuple<int> index, tuple<int> shape);
  tuple<int> c2idx(int i, tuple<int> shape);
  void make_launch_parameters(tuple<int> shape, int dim, dim3 *bdim, dim3 *gdim);
}

#include <kernels.h>
#include <tuple.h>
#include <matrix.h>
