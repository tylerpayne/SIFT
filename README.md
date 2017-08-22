# chai
chai is a CUDA accelerated matrix library with a focus on computer vision methods.

# usage
a basic example:
````C
#include <matrix.h>
    
int main(int argc, char **argv)
{
  init_cuda_libs(); // must call before using chai
      
  Shape s1 = {10,20}; 
  Shape s2 = {20,10};
  Shape s3 = {10,10};
      
  Matrix *m1, *m2, *m3;
  new_empty_matrix(&m1,s1);
  new_empty_matrix(&m2,s2);
  new_empty_matrix(&m3,s3);
      
  float a=0.5, b=0.5;
      
  fill(m1,a);
  fill(m2,b);

  mdot(m1,m2,m3);
      
  print_matrix(m3,"chai");
  
  free_matrix(m1);
  free_matrix(m2);
  free_matrix(m3);
  
  destroy_cuda_libs(); // best practice
  return 0;
}
````

# reference

Matrix Initialization

````C
void new_empty_matrix(Matrix **new_matrix, Shape shape);
void new_device_matrix(Matrix **new_matrix, float* dev_ptr, Shape shape);
void new_host_matrix(Matrix **new_matrix, float* host_ptr, Shape shape);
int free_matrix(Matrix *m);

//*a must be initialized
int fill(Matrix *a, float b);
int linfill(Matrix *a, float from, float to);
int rand_uniform(Matrix *out);
````

Matrix Indexing & Copying

````C
float get_element(Matrix *a, Point2 id);
int copy_region(Matrix *a, Matrix *out, Point2 a_idx, Point2 out_idx, Shape shape);
int copy(Matrix *a, Matrix *out, Rect region);
````

Arithmetic

````C
int add(Matrix *a, Matrix *b, Matrix *out);
int addc(Matrix *a, float b, Matrix *out);
int subtract(Matrix *a, Matrix *b, Matrix *out);
int subtractc(Matrix *a, float b, Matrix *out);
int multiply(Matrix *a, Matrix *b, Matrix *out);
int multiplyc(Matrix *a, float b, Matrix *out);
int divide(Matrix *a, Matrix *b, Matrix *out);
int dividec(Matrix *a, float b, Matrix *out);
````

Matrix Operations

````C
void mdot(Matrix *a, Matrix *b, Matrix *out);
int euclid_norm(Matrix *a, float *out);
int sum(Matrix *a, float *out);
````

Logic

````C
int gt(Matrix *a, Matrix *b, Matrix *out);
int gte(Matrix *a, Matrix *b, Matrix *out);
int lt(Matrix *a, Matrix *b, Matrix *out);
int lte(Matrix *a, Matrix *b, Matrix *out);
int eq(Matrix *a, Matrix *b, Matrix *out);
````

Trig

````C
int mcos(Matrix *a, Matrix *out);
int msin(Matrix *a, Matrix *out);
int mtan(Matrix *a, Matrix *out);
int macos(Matrix *a, Matrix *out);
int masin(Matrix *a, Matrix *out);
int matan(Matrix *a, Matrix *out);
int matan2(Matrix *a, Matrix *b, Matrix *out);
int mhypot(Matrix *a, Matrix *b, Matrix *out);
````

Math etc

````C
int msqrt(Matrix *a, Matrix *out);
int mabs(Matrix *a, Matrix *out);
int mexp(Matrix *a, Matrix *out);
int mlog(Matrix *a, Matrix *out);
int mpow(Matrix *a, float e, Matrix *out);
````

Statistics

````C
int argmax(Matrix *a, int *index);
int argmin(Matrix *a, int *index);
int count_in_range(Matrix *a, float from, float to, int *count);
````

Image

````C
int load_image(Matrix **out, const char *filepath);
int write_image(Matrix *im, const char *filepath);


int imgrad(Matrix *src, Matrix* dX, Matrix* dY, Matrix* mag, Matrix* angle);
int resample(Matrix *a, Matrix *out, Shape shape);
int dog(Matrix *a, Matrix *out, Shape kernel_shape, float std1, float std2); // differnce of gaussian
````

Signal
````C
int convolve(Matrix* a, Matrix* b, Matrix* out);
````

### disclaimer
chai is an academic project of mine and is therefore unstable and constantly under development. I do not claim it to be production viable.
