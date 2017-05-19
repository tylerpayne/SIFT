#include <matrix.h>

extern cudaStream_t _cudaStream;

float get_element(Matrix *a, Point2 id)
{
  int c = IDX2C(id,a->shape);
  if (a->isHostSide)
    return a->host_ptr[c];
  else
  {
    float retval;
    cuda_safe_call(
      cudaMemcpy(
          &retval, a->dev_ptr + c,
          sizeof(float), cudaMemcpyDeviceToHost
      )
    );
    return retval;
  }
}
