// LocalMax
__global__ void LocalMaxKernel(float* pSrc, float* pDst, NppiSize oSize, int windowWidth)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int offset = (y*windowWidth*oSize.width) + (x*windowWidth);
  int row = (offset/oSize.width);
  int maxIdx = offset;
  float maxVal = pSrc[offset];
  int totalLength = oSize.width*oSize.height;
  for (int i = 0; i < windowWidth; i ++)
  {
    for (int j = 0; j < windowWidth; j++)
    {
      int srcOffset = (row*oSize.width) + (oSize.width*i) + (x*windowWidth) + j;
      if (srcOffset < totalLength)
      {
        float thisVal = pSrc[srcOffset];
        if (thisVal > maxVal)
        {
          maxVal = thisVal;
          maxIdx = srcOffset;
        }
      }
    }
  }
  pDst[maxIdx] = maxVal;
}

__global__ void LocalMaxIdxKernel(float* pSrc, float* pDst, int* pIdx, NppiSize oSize, int windowWidth)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int offset = (y*windowWidth*oSize.width) + (x*windowWidth);
  int row = (offset/oSize.width);
  int maxIdx = offset;
  float maxVal = pSrc[offset];
  int totalLength = oSize.width*oSize.height;
  for (int i = 0; i < windowWidth; i ++)
  {
    for (int j = 0; j < windowWidth; j++)
    {
      int srcOffset = (row*oSize.width) + (oSize.width*i) + (x*windowWidth) + j;
      if (srcOffset < totalLength)
      {
        float thisVal = pSrc[srcOffset];
        if (thisVal > maxVal)
        {
          maxVal = thisVal;
          maxIdx = srcOffset;
        }
      }
    }
  }
  pDst[maxIdx] = maxVal;
  pIdx[IDX2CKernel(y,x,1)] = maxIdx;
}
