#include <matrix.h>
#include <curand.h>

#ifdef __cplusplus
extern "C" {
#endif

int uniform(Matrix *out, Shape shape)
{
  memassert(out,DEVICE);
  float *dev_ptr;
  cuda_safe_call(cudaMalloc(&dev_ptr,shape.width*shape.height*sizeof(float)));
  curandGenerator_t gen;
  curandCreateGenerator(&gen,
                CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen,
                1234ULL);
  curandGenerateUniform(gen, dev_ptr, shape.width*shape.height);
  out->dev_ptr = dev_ptr;
  return 0;
}

#ifdef __cplusplus
}
#endif
