#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;

int get_features(Matrix *a, Matrix *features, int *index_array, int len, Shape feature_shape)
{
  memassert(a,DEVICE);
  memassert(features,DEVICE);

  for (int i = 0; i < len; i++)
  {
    Point2 src_idx = C2IDX(index_array[i],a->shape);
    Point2 dest_idx = {0,i*feature_shape.height};
    copy_region(a,features,src_idx,dest_idx,feature_shape);
  }

  return 0;
}

#ifdef __cplusplus
}
#endif
