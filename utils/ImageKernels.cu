__global__ void DownsampleKernel(float* A, float* B, int ald, int atd, int bld, int btd)
{
  int row = (blockIdx.x * blockDim.x) + threadIdx.x;
  int col = (blockIdx.y * blockDim.y) + threadIdx.y;

  float scaleX = ((float)ald) / ((float)bld);
  float scaleY = ((float)atd) / ((float)btd);

  if (row < bld && col < btd)
  {
    float xn = ((float)row) * scaleX;
    float yn = ((float)col) * scaleY;

    float x1 = (float)floor(xn);
    float x2 = (float)ceil(xn);

    float y1 = (float)floor(yn);
    float y2 = (float)ceil(yn);

    float y2n = y2 - yn;
    float yn1 = yn - y1;
    float x2n = x2-xn;
    float xn1 = xn-x1;
    //float y21 = y2 - y1;
    //float x21 = x2 - x1;

    if (y2n == 0 && yn1 == 0)
    {
      B[IDX2CKernel(row,col,btd)] = A[IDX2CKernel((int)xn,(int)yn,atd)];
    } else
    {
      float J = (y2n*A[IDX2CKernel((int)x1,(int)y1,atd)]) + (yn1*A[IDX2CKernel((int)x1,(int)y2,atd)]);
      float K = (y2n*A[IDX2CKernel((int)x2,(int)y1,atd)]) + (yn1*A[IDX2CKernel((int)x2,(int)y2,atd)]);
      if (x2n == 0 && xn1 == 0)
      {
        B[IDX2CKernel(row,col,btd)] = A[IDX2CKernel((int)xn,(int)yn,atd)];
      } else
      {
        B[IDX2CKernel(row,col,btd)] = (x2n*J) + (xn1*K);
      }

    }

  }
}
