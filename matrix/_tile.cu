/*#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;

int tile(Matrix *a, Matrix *b, Matrix *out, Shape window, int (*op)(Matrix*,Matrix*,Matrix*))
{
  memassert(a, DEVICE);
  if (b != NULL) memassert(b, DEVICE);

  Matrix *ret;
  if (out != NULL) ret = out;
  else ret = a;

  memassert(ret, DEVICE);

  int w_steps, w_step, h_steps, h_step;
  w_step = window.width;
  h_step = window.height;
  w_steps = a->shape.width / w_step;
  h_steps = a->shape.height / h_step;

  printf("Tiling across %i streams\n",w_steps*h_steps);
  cudaStream_t old = _cudaStream, current[w_steps*h_steps];

  int iter = 0;
  Shape wh = {w_step,h_step};

  for (int i = 0; i <h_steps*h_step; i+= h_step)
  {
    for (int j = 0; j < w_steps*w_step; j+= w_step)
    {
      cuda_safe_call(cudaStreamCreateWithFlags(&current[iter],cudaStreamNonBlocking));
      set_stream(current[iter]);

      /*Matrix *a_slice, *b_slice, *ret_slice;
      new_empty_matrix(&a_slice,wh);
      new_empty_matrix(&b_slice,wh);
      new_empty_matrix(&ret_slice,wh);
      a_slice->isHostSide = FALSE;
      b_slice->isHostSide = FALSE;
      ret_slice->isHostSide = FALSE;

      copy_region(a,a_slice,{j,i},{0,0},wh);
      copy_region(b,b_slice,{j,i},{0,0},wh);
      copy_region(ret,ret_slice,{j,i},{0,0},wh);

      op({j,i},a,b,ret);

      /*copy_region(a_slice,a,{0,0},{j,i},wh);
      copy_region(b_slice,b,{0,0},{j,i},wh);
      copy_region(ret_slice,ret,{0,0},{j,i},wh);
      free_matrix(a_slice);
      free_matrix(b_slice);
      free_matrix(ret_slice)
      iter++;
    }
  }
  cuda_safe_call(cudaDeviceSynchronize());
  for (int i = 0; i < h_steps*w_steps; i++)
  {
    cuda_safe_call(cudaStreamDestroy(current[i]));
  }
  set_stream(old);
  return 0;
}

#ifdef __cplusplus
}
#endif*/
