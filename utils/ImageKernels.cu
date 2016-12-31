__device__ int IDX2CKernel(int i, int j, int td)
{
  return (i*td)+j;
}

// LocalMax
__global__ void LocalMaxKernel(Npp32f* pSrc, Npp32f* pDst, NppiSize oSize, int windowWidth)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int offset = (y*windowWidth*oSize.width) + (x*windowWidth);
  int row = (offset/oSize.width);
  int maxIdx = offset;
  Npp32f maxVal = pSrc[offset];
  int totalLength = oSize.width*oSize.height;
  for (int i = 0; i < windowWidth; i ++)
  {
    for (int j = 0; j < windowWidth; j++)
    {
      int srcOffset = (row*oSize.width) + (oSize.width*i) + (x*windowWidth) + j;
      if (srcOffset < totalLength)
      {
        Npp32f thisVal = pSrc[srcOffset];
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

__global__ void LocalMaxIdxKernel(Npp32f* pSrc, Npp32f* pDst, int* pIdx, NppiSize oSize, int windowWidth)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int offset = (y*windowWidth*oSize.width) + (x*windowWidth);
  int row = (offset/oSize.width);
  int maxIdx = offset;
  Npp32f maxVal = pSrc[offset];
  int totalLength = oSize.width*oSize.height;
  for (int i = 0; i < windowWidth; i ++)
  {
    for (int j = 0; j < windowWidth; j++)
    {
      int srcOffset = (row*oSize.width) + (oSize.width*i) + (x*windowWidth) + j;
      if (srcOffset < totalLength)
      {
        Npp32f thisVal = pSrc[srcOffset];
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
