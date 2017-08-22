#include <matrix.h>
#include <lodepng.h>

#ifdef __cplusplus
extern "C" {
#endif

int load_image(Matrix **out, const char *filepath)
{
  unsigned error;
  unsigned char* image;
  unsigned width, height;

  error = lodepng_decode32_file(&image, &width, &height, filepath);
  if(error) printf("read error %u: %s\n", error, lodepng_error_text(error));

  float *h_data = (float*)malloc(sizeof(float)*width*height);
  Shape shape = {(int)width,(int)height};
  for (int i = 0; i < SHAPE2LEN(shape); i++)
  {
    float accum = (float)image[i*4];
    accum += (float)image[(i*4)+1];
    accum += (float)image[(i*4)+2];
    //accum += (float)image[(i*4)+3];
    h_data[i] = accum/3.0;
  }
  new_host_matrix(out,h_data,shape);
  dividec(*out,255.0,NULL);
  free(image);
}

int write_image(Matrix *im, const char *filepath)
{
  cuda_safe_call(cudaDeviceSynchronize());
  memassert(im,HOST);
  unsigned char *image = (unsigned char*)malloc(SHAPE2LEN(im->shape)*4);
  for (int i = 0; i < SHAPE2LEN(im->shape); i++)
  {
    unsigned char p = (unsigned char)get_element(im,C2IDX(i,im->shape));
    image[i*4] = p;
    image[i*4 + 1] = p;
    image[i*4 + 2] = p;
    image[i*4 + 3] = 255;
  }
  unsigned error = lodepng_encode32_file(filepath, image, im->shape.width, im->shape.height);
  if(error) printf("write error %u: %s\n", error, lodepng_error_text(error));
  free(image);
  //multiplyc(im,(1.0/255.0)*max,NULL);

}

#ifdef __cplusplus
}
#endif
