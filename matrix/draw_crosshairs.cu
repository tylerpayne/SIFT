#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;

int draw_crosshairs(Matrix *a, int *index_array, int len, int cross_width, float brightness)
{
  memassert(a, DEVICE);

  Matrix *ones, *crosshair,*tmp;
  new_empty_matrix(&ones,{cross_width,cross_width});
  new_empty_matrix(&crosshair,{cross_width,cross_width});
  new_empty_matrix(&tmp,a->shape);

  fill(ones,1.0);

  copy_region(ones,crosshair,{0,0},{0,cross_width/2},{cross_width,1});
  copy_region(ones,crosshair,{0,0},{cross_width/2,0},{1,cross_width});
  free_matrix(ones);
  multiplyc(crosshair,brightness,NULL);
  for (int i = 0; i < len; i++)
  {
    fill(tmp,0);
    Point2 src_idx = {0,0};
    Point2 dest_idx = {C2IDX(index_array[i],a->shape)};
    copy_region(crosshair,tmp,src_idx,dest_idx,{cross_width,cross_width});
    add(a,tmp,a);
  }
  free_matrix(tmp);
  free_matrix(crosshair);

  return 0;
}

#ifdef __cplusplus
}
#endif
