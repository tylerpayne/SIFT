#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern curandGenerator_t _curandGenerator;

int rand_uniform(Matrix *out)
{
  memassert(out,DEVICE);
  float *dev_ptr;
  Shape shape = out->shape;
  cuda_safe_call(
    cudaMalloc(&dev_ptr,
      shape.width*shape.height*sizeof(float)
    )
  );

  curand_safe_call(
    curandGenerateUniform(_curandGenerator,
       dev_ptr, shape.width*shape.height
     )
   );

  out->dev_ptr = dev_ptr;
  return 0;
}

#ifdef __cplusplus
}
#endif
