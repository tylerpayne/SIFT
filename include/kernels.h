namespace chai
{
  namespace cuda
  {

    __device__   int tuple_product(int_tuple shape);
    __device__   void c2idx_kernel(int i, int shape[2], int *r, int *c);
    __device__   int idx2c_kernel(int index[2], int shape[2]);

    // ARITHMETIC
    template<typename K>
    __global__ void add_kernel(K *a, K *b, K *c, int len);

    template<typename K>
    __global__   void addc_kernel(K *a, K b, K *c, int len);

    template<typename K>
    __global__   void divide_kernel(K *a, K *b, K *c, int len);

    template<typename K>
    __global__   void dividec_kernel(K *a, K b, K *c, int len);

    template<typename K>
    __global__   void multiply_kernel(K *a, K *b, K *c, int len);

    template<typename K>
    __global__   void multiplyc_kernel(K *a, K b, K *c, int len);

    template<typename K>
    __global__   void subtract_kernel(K *a, K *b, K *c, int len);

    template<typename K>
    __global__   void subtractc_kernel(K *a, K b, K *c, int len);

    // MATH ETC

    template<typename K>
    __global__   void sqrt_kernel(K *a, K *b, int len);

    template<typename K>
    __global__   void abs_kernel(K *a, K *b, int len);

    template<typename K>
    __global__   void exp_kernel(K *a, K *b, int len);

    template<typename K>
    __global__   void log_kernel(K *a, K *b, int len);

    template<typename K>
    __global__   void pow_kernel(K *a, K b, K *c, int len);

    ///////////////////////
    // MATRIX OPERATIONS //
    //////////////////////

    template<typename K>
    matrix<K> dot_kernel(matrix<K> &a, matrix<K> &b);

  }
}
