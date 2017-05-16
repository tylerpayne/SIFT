#ifdef __cplusplus
extern "C" {
#endif

//HELPER_FUNCS

int IDX2C(Point2 index, Shape shape);
Point2 C2IDX(int i, Shape shape);
int SHAPE2LEN(Shape shape);
void make_launch_parameters(Shape shape, int dim, dim3 *bdim, dim3 *gdim);
int print_matrix(Matrix *a, const char *msg);

//KERNEL_HELPER_FUNCS

__device__ int IDX2C_kernel(Point2 index, Shape shape);
__device__ Point2 C2IDX_kernel(int i, Shape shape);
__device__ int SHAPE2LEN_kernel(Shape shape);

//CUDA_SAFE_CALLS

void cublas_safe_call(cublasStatus_t err);
void cuda_safe_call(cudaError_t err);
void npp_safe_call(NppStatus err);

//MATRIX INIT

void new_empty_matrix(Matrix **new_matrix, Shape shape);
void new_device_matrix(Matrix **new_matrix, float* dev_ptr, Shape shape);
void new_host_matrix(Matrix **new_matrix, float* host_ptr, Shape shape);
int free_matrix(Matrix *m);

//INDEXING

__global__ void get_region_kernel(float *a, float *b, Shape shape, Rect region);

float get_element(Matrix *a, Point2 id);
int copy(Matrix *a, Matrix *out, Rect region);


//MEMCHECK

int memassert(Matrix *m, int dest);

//ARITHMETIC

__global__ void add_kernel(float *a, float *b, float *c, Shape shape);
__global__ void subtract_kernel(float *a, float *b, float *c, Shape shape);
__global__ void multiply_kernel(float *a, float *b, float *c, Shape shape);
__global__ void divide_kernel(float *a, float *b, float *c, Shape shape);

int add(Matrix *a, Matrix *b, Matrix *out);
int subtract(Matrix *a, Matrix *b, Matrix *out);
int multiply(Matrix *a, Matrix *b, Matrix *out);
int divide(Matrix *a, Matrix *b, Matrix *out);

//LINEAR ALGEBRA

void dot(Matrix *a, Matrix *b, Matrix *out);

//LOGIC

__global__ void gt_kernel(float *a, float *b, float *c, Shape shape);
__global__ void gte_kernel(float *a, float *b, float *c, Shape shape);
__global__ void lt_kernel(float *a, float *b, float *c, Shape shape);
__global__ void lte_kernel(float *a, float *b, float *c, Shape shape);
__global__ void eq_kernel(float *a, float *b, float *c, Shape shape);

int gt(Matrix *a, Matrix *b, Matrix *out);
int gte(Matrix *a, Matrix *b, Matrix *out);
int lt(Matrix *a, Matrix *b, Matrix *out);
int lte(Matrix *a, Matrix *b, Matrix *out);
int eq(Matrix *a, Matrix *b, Matrix *out);

//TRIG

__global__ void cos_kernel(float *a, float *b, Shape shape);
__global__ void sin_kernel(float *a, float *b, Shape shape);
__global__ void tan_kernel(float *a, float *b, Shape shape);
__global__ void acos_kernel(float *a, float *b, Shape shape);
__global__ void asin_kernel(float *a, float *b, Shape shape);
__global__ void atan_kernel(float *a, float *b, Shape shape);
__global__ void atan2_kernel(float *a, float *b, float *c, Shape shape);
__global__ void hypot_kernel(float *a, float *b, float *c, Shape shape);


int mcos(Matrix *a, Matrix *out);
int msin(Matrix *a, Matrix *out);
int mtan(Matrix *a, Matrix *out);
int macos(Matrix *a, Matrix *out);
int masin(Matrix *a, Matrix *out);
int matan(Matrix *a, Matrix *out);
int matan2(Matrix *a, Matrix *b, Matrix *out);
int mhypot(Matrix *a, Matrix *b, Matrix *out);

//MATH_ETC

__global__ void sqrt_kernel(float *a, float *b, Shape shape);
__global__ void abs_kernel(float *a, float *b, Shape shape);
__global__ void exp_kernel(float *a, float *b, Shape shape);
__global__ void log_kernel(float *a, float *b, Shape shape);
__global__ void pow_kernel(float *a, float b, float *c, Shape shape);

int msqrt(Matrix *a, Matrix *out);
int mabs(Matrix *a, Matrix *out);
int mexp(Matrix *a, Matrix *out);
int mlog(Matrix *a, Matrix *out);
int mpow(Matrix *a, float e, Matrix *out);

//SIGNAL

int convolve(Matrix* a, Matrix* b, Matrix* out);

//IMAGE

int imgrad(Matrix* src, Matrix* dX, Matrix* dY, Matrix* mag, Matrix* angle);
int resample(Matrix* a, Matrix* out, Shape shape);

// UTILITY

int generateGaussian(Matrix *a, Shape shape, float w, float h);

// RAND
int uniform(Matrix *out, Shape shape);


#ifdef __cplusplus
}
#endif