namespace chai
{
  namespace cuda
  {
    typedef struct int_tuple
    {
      int *components, length;
    } int_tuple;

    __device__ int tuple_product(int_tuple shape);
    __device__ void c2idx_kernel(int i, int shape[2], int *r, int *c);
    __device__ int idx2c_kernel(int index[2], int shape[2]);

    // ARITHMETIC
    template<typename K>
    __global__ void add_kernel(K *a, K *b, K *c, int_tuple shape);

    template<typename K>
    __global__ void addc_kernel(K *a, K b, K *c, int_tuple shape);

    template<typename K>
    __global__ void divide_kernel(K *a, K *b, K *c, int_tuple shape);

    template<typename K>
    __global__ void dividec_kernel(K *a, K b, K *c, int_tuple shape);

    template<typename K>
    __global__ void multiply_kernel(K *a, K *b, K *c, int_tuple shape);

    template<typename K>
    __global__ void multiplyc_kernel(K *a, K b, K *c, int_tuple shape);

    template<typename K>
    __global__ void subtract_kernel(K *a, K *b, K *c, int_tuple shape);

    template<typename K>
    __global__ void subtractc_kernel(K *a, K b, K *c, int_tuple shape);

    // MATH ETC

    template<typename K>
    __global__ void sqrt_kernel(K *a, K *b, int_tuple shape);

    template<typename K>
    __global__ void abs_kernel(K *a, K *b, int_tuple shape);

    template<typename K>
    __global__ void exp_kernel(K *a, K *b, int_tuple shape);

    template<typename K>
    __global__ void log_kernel(K *a, K *b, int_tuple shape);

    template<typename K>
    __global__ void pow_kernel(K *a, K b, K *c, int_tuple shape);
  }
}
