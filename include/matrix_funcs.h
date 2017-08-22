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

//CUDA_LIB_CALLS

int init_cuda_libs();
int destroy_cuda_libs();
int set_stream(cudaStream_t stream);

//CUDA_SAFE_CALLS

void cublas_safe_call(cublasStatus_t err);
void cuda_safe_call(cudaError_t err);
void npp_safe_call(NppStatus err);
void curand_safe_call(curandStatus_t err);

//MATRIX INIT

void new_empty_matrix(Matrix **new_matrix, Shape shape);
void new_device_matrix(Matrix **new_matrix, float* dev_ptr, Shape shape);
void new_host_matrix(Matrix **new_matrix, float* host_ptr, Shape shape);
int free_matrix(Matrix *m);

//INDEXING

__global__ void copy_region_kernel(float *a, float *b, Point2 a_idx, Point2 b_idx, Shape region, Shape a_shape, Shape b_shape);

float get_element(Matrix *a, Point2 id);
int copy_region(Matrix *a, Matrix *out, Point2 a_idx, Point2 out_idx, Shape shape);
int copy(Matrix *a, Matrix *out, Rect region);

__global__ void fill_kernel(float *a, float b, Shape shape);
int fill(Matrix *a, float b);

__global__ void linfill_kernel(float *a, float from, float step, Shape shape);
int linfill(Matrix *a, float from, float to);

//MEMCHECK

int memassert(Matrix *m, int dest);

//tile

int tile(Matrix *a, Matrix *b, Matrix *out, Shape window, int (*op)(Matrix*,Matrix*,Matrix*));

//ARITHMETIC

__global__ void add_kernel(float *a, float *b, float *c, Shape shape);
__global__ void addc_kernel(float *a, float b, float *c, Shape shape);
__global__ void subtract_kernel(float *a, float *b, float *c, Shape shape);
__global__ void subtractc_kernel(float *a, float b, float *c, Shape shape);
__global__ void multiply_kernel(float *a, float *b, float *c, Shape shape);
__global__ void multiplyc_kernel(float *a, float b, float *c, Shape shape);
__global__ void divide_kernel(float *a, float *b, float *c, Shape shape);
__global__ void dividec_kernel(float *a, float b, float *c, Shape shape);


int add(Matrix *a, Matrix *b, Matrix *out);
int addc(Matrix *a, float b, Matrix *out);
int subtract(Matrix *a, Matrix *b, Matrix *out);
int subtractc(Matrix *a, float b, Matrix *out);
int multiply(Matrix *a, Matrix *b, Matrix *out);
int multiplyc(Matrix *a, float b, Matrix *out);
int divide(Matrix *a, Matrix *b, Matrix *out);
int dividec(Matrix *a, float b, Matrix *out);

//VECTOR + MATRIX OPs

void mdot(Matrix *a, Matrix *b, Matrix *out);
int euclid_norm(Matrix *a, float *out);
int sum(Matrix *a, float *out);

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

//STATISTICS

int argmax(Matrix *a, int *index);
int argmin(Matrix *a, int *index);
int count_in_range(Matrix *a, float from, float to, int *count);

__global__ void range_reduce_kernel(float *a, float from, float to, int *d_index, Shape shape);
int range_reduce(Matrix *a, float from, float to, int **index_array, int *len);

int histogram_range(Matrix *a, Matrix *ranges, int *d_histogram);
//IMAGE

int imgrad(Matrix *src, Matrix* dX, Matrix* dY, Matrix* mag, Matrix* angle);
int resample(Matrix *a, Matrix *out, Shape shape);
int dilate(Matrix *a, unsigned char *d_b, Shape b_shape, Matrix *out);

int dog(Matrix *a, Matrix *out, Shape kernel_shape, float std1, float std2);
int draw_crosshairs(Matrix *a, int *index_array, int len, int cross_width, float brightness);
int get_features(Matrix *a, Matrix *features, int *index_array, int len, Shape feature_shape);

// UTILITY

int generate_gaussian(Matrix **out, Shape shape, float w, float h);

// RAND
int rand_uniform(Matrix *out);

// IMIO
int load_image(Matrix **out, const char *filepath);
int write_image(Matrix *im, const char *filepath);

#ifdef __cplusplus
}
#endif
