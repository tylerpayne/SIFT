#include <matrix.h>

int extract_corner_features(Matrix *img, Matrix **features, int kernel_width, float s1, float s2)
{
  Matrix *angle, *d2, *o;
  new_empty_matrix(&o,img->shape);
  new_empty_matrix(&d2,img->shape);
  new_empty_matrix(&angle,img->shape);
  imgrad(img,NULL,NULL,NULL,angle);

  Shape kernel_shape = {kernel_width,kernel_width};
  dog(img,d2,kernel_shape,s1,s2);

  //mabs(d2,NULL);

  int index;
  float maxval;
  argmax(d2,&index);
  Point2 idx = C2IDX(index,img->shape);
  maxval = get_element(d2,idx);

  fill(o,maxval);
  eq(d2,o,NULL);

  int *reduced, count;
  range_reduce(d2,0.9,1.1, &reduced, &count);

  draw_crosshairs(img,reduced,count,9,400.0);

  new_empty_matrix(features,{16,16*count});
  Shape region = {16,16};

  get_features(angle,*features,reduced,count,region);

  (*features)->shape = {16*16,count};

  free_matrix(angle);
  free_matrix(d2);
  free_matrix(o);
  free(reduced);

  return 0;
}

int histogram_features(Point2 tile_id, Matrix *features, Matrix *b, Matrix *ret)
{
  int nbins = features->shape.width + 1;
  Matrix *ranges;
  new_empty_matrix(&ranges,{nbins,1});

  linfill(ranges,-2.0*M_PI,2.0*M_PI);

  int *histogram;
  cuda_safe_call(
    cudaMalloc(&histogram,sizeof(int)*(nbins-1))
  );

  histogram_range(features,ranges,histogram);



}

int main(int argc, char **argv)
{
  init_cuda_libs();
  int w = (argc > 1) ? atoi(argv[1]) : 9;
  float s1 = (argc > 2) ? atof(argv[2]) : 5;
  float s2 = (argc > 3) ? atof(argv[3]) : 9;

  Matrix *l_large, *r_large, *l, *r;
  load_image(&l_large,"left.png");
  load_image(&r_large,"right.png");

  int divisor = 8;

  Shape small = {l_large->shape.width/divisor,l_large->shape.height/divisor};
  new_empty_matrix(&l,small);
  new_empty_matrix(&r,small);
  resample(l_large,l,small);
  resample(r_large,r,small);
  multiplyc(l,255.0,l);
  multiplyc(r,255.0,r);

  Matrix *l_features,*r_features, *cost;
  extract_corner_features(l,&l_features,w,s1,s2);
  extract_corner_features(r,&r_features,w,s1,s2);

  new_empty_matrix(&cost,{l_features->shape.height,r_features->shape.height});

  r_features->T = TRUE;
  mdot(l_features,r_features,cost);
  r_features->T = FALSE;

  write_image(cost, "cost.png");

  write_image(l_features,"l_feat.png");
  write_image(l,"l1.png");

  write_image(r_features,"r_feat.png");
  write_image(r,"r1.png");

  free_matrix(cost);
  free_matrix(l_large);
  free_matrix(r_large);
  free_matrix(l);
  free_matrix(r);
  free_matrix(l_features);
  free_matrix(r_features);

  destroy_cuda_libs();
  return 0;
}
