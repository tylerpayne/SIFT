namespace chai {

namespace cuda {
void init_cuda_libs();
void destroy_cuda_libs();
void set_stream(cudaStream_t s);

template <typename K> void safe_call(K err);

void cublas_safe_call(cublasStatus_t err);
void cuda_safe_call(cudaError_t err);
void npp_safe_call(NppStatus err);
void curand_safe_call(curandStatus_t err);

__device__ int tuple_product(tuple<int> shape);
__device__ void c2idx_kernel(int i, int shape[2], int *r, int *c);
__device__ int idx2c_kernel(int index[2], int shape[2]);
}

}
