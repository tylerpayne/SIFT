#include <matrix.h>

int generate_gaussian(Matrix **out, Shape shape, float w, float h)
{

  float *g = (float*)malloc(sizeof(float)*shape.width*shape.height);
  int w_radius = shape.width/2;
  int h_radius = shape.height/2;
  for (int y = 0; y < shape.height; y++)
  {
    for (int x = 0; x < shape.width; x++)
    {
      Point2 p = {x,y};
      g[IDX2C(p,shape)] = (((x-w_radius)*(x-w_radius))/(2*(w*w))) + (((y-h_radius)*(y-h_radius))/(2*(h*h)));
    }
  }
  new_host_matrix(out,g,shape);
  multiplyc(*out,-1.0,NULL);
  mexp(*out,NULL);
  float norm;
  euclid_norm(*out,&norm);
  dividec(*out,norm,NULL);
  return 0;
}
