#include <matrix.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cudaStream_t _cudaStream;

int range_reduce(Matrix *a, float from, float to, int **index_array, int *len)
{
  memassert(a, DEVICE);

  int count;
  count_in_range(a,from,to,&count);
  *len = count;

  int *d_index;
  size_t d_size = sizeof(int)*SHAPE2LEN(a->shape);
  cuda_safe_call(cudaMalloc((void **)&d_index,d_size));
  cuda_safe_call(cudaMemset(d_index,0,d_size));

  dim3 bdim, gdim;
  make_launch_parameters(a->shape,1,&bdim,&gdim);
  range_reduce_kernel<<<gdim,bdim,sizeof(int)*bdim.x,_cudaStream>>>(a->dev_ptr,from,to,d_index,a->shape);

  int *h_array = (int*)malloc(d_size);
  cuda_safe_call(
    cudaMemcpyAsync(h_array,d_index,d_size,cudaMemcpyDeviceToHost,_cudaStream)
  );

  cuda_safe_call(cudaDeviceSynchronize());

  *index_array = (int*)malloc(sizeof(int)*count);
  int iter = 0;
  for (int i = 0; i < (gdim.x-1)*bdim.x; i+=bdim.x)
  {
    int nonzero_count = h_array[i];
    for (int j = 1; j < nonzero_count; j++)
    {
      (*index_array)[iter++] = h_array[i+j];
      if (iter >= SHAPE2LEN(a->shape)) break;
    }
    if (iter >= SHAPE2LEN(a->shape)) break;
  }

  free(h_array);

  cuda_safe_call(
    cudaFree(d_index)
  );

  return 0;
}

#ifdef __cplusplus
}
#endif
