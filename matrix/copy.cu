#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

int copy(Matrix *a, Matrix *out, Rect region)
{
  memassert(a, DEVICE);
  memassert(out, DEVICE);
  for (int y = region.origin.y; y < region.origin.y + region.shape.height; y++)
  {
    cuda_safe_call(cudaMemcpy(
      out->dev_ptr+(y-region.origin.y)*region.shape.width,
      a->dev_ptr+(y-region.origin.y)*a->shape.width,
      sizeof(float)*region.shape.width,
      cudaMemcpyDeviceToDevice
    ));
  }
  return 0;
}

#ifdef __cplusplus
}
#endif
