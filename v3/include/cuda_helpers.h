#ifndef _CORE_H_
#define _CORE_H_

//CUDA_LIB_CALLS

int init_cuda_libs();
int destroy_cuda_libs();
int set_stream(cudaStream_t stream);

//CUDA_SAFE_CALLS

void cublas_safe_call(cublasStatus_t err);
void cuda_safe_call(cudaError_t err);
void npp_safe_call(NppStatus err);
void curand_safe_call(curandStatus_t err);

#endif
