#include <math_constants.h>

__device__ int IDX2CKernel(int i, int j, int td)
{
  return (i*td)+j;
}
// LocalMax
__global__ void LocalMaxKernel(float* pSrc, float* pDst, NppiSize oSize, int windowWidth)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int maxLength = oSize.width*oSize.height;
  int row = y*windowWidth;
  int col = x*windowWidth;
  int offset = IDX2CKernel(row,col,oSize.width);
  if (offset < maxLength)
  {
    int maxIdx = offset;
    float maxVal = pSrc[offset];
    for (int i = 0; i < windowWidth; i++)
    {
      for (int j = 0; j < windowWidth; j++)
      {
        int localOffset = IDX2CKernel(row+i,col+j,oSize.width);
        if (pSrc[localOffset] > maxVal && localOffset < maxLength)
        {
          maxVal = pSrc[localOffset];
          maxIdx = localOffset;
        }
      }
    }
    if (maxIdx != offset)
    {
      pDst[maxIdx] = maxVal;
    } else
    {
      pDst[maxIdx] = 0;
    }
  }
}

__global__ void LocalMaxIdxKernel(float* pSrc, float* pDst, int* pIdx, NppiSize oSize, int windowWidth)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int maxLength = oSize.width*oSize.height;
  int row = y*windowWidth;
  int col = x*windowWidth;
  int offset = IDX2CKernel(row,col,oSize.width);
  if (offset < maxLength)
  {
    int maxIdx = offset;
    float maxVal = pSrc[offset];
    for (int i = 0; i < windowWidth; i++)
    {
      for (int j = 0; j < windowWidth; j++)
      {
        int localOffset = IDX2CKernel(row+i,col+j,oSize.width);
        if (pSrc[localOffset] > maxVal && localOffset < maxLength)
        {
          maxVal = pSrc[localOffset];
          maxIdx = localOffset;
        }
      }
    }
    if (maxIdx != offset)
    {
      pIdx[IDX2CKernel(y,x,windowWidth)] = maxIdx;
      pDst[maxIdx] = maxVal;
    } else
    {
      pIdx[IDX2CKernel(y,x,windowWidth)] = -1;
      pDst[maxIdx] = 0;
    }
  }
}


//Local Contrast
__global__ void LocalContrastKernel(float* pSrc, float* pDst, NppiSize oSize, int windowWidth)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int maxLength = oSize.width*oSize.height;
  int row = y*windowWidth;
  int col = x*windowWidth;
  int offset = IDX2CKernel(row,col,oSize.width);
  if (offset < maxLength)
  {
    float maxVal = pSrc[offset];
    float minVal = pSrc[offset];
    for (int i = 0; i < windowWidth; i++)
    {
      for (int j = 0; j < windowWidth; j++)
      {
        int localOffset = IDX2CKernel(row+i,col+j,oSize.width);
        if (pSrc[localOffset] > maxVal && localOffset < maxLength)
        {
          maxVal = pSrc[localOffset];
        }
        if (pSrc[localOffset] < maxVal && localOffset < maxLength)
        {
          minVal = pSrc[localOffset];
        }
      }
    }
    float contrast = -1;
    if (maxVal != 0)
    {
      contrast = (maxVal-minVal)/maxVal;
    }
    for (int i = 0; i < windowWidth; i++)
    {
      for (int j = 0; j < windowWidth; j++)
      {
        int localOffset = IDX2CKernel(row+i,col+j,oSize.width);
        if (localOffset < maxLength)
        {
          pDst[localOffset] = contrast;
        }
      }
    }
  }
}

//UNORIENT FEATURE
__global__ void UnorientFeatureKernel(float* features, int nFeatures, int nDim, int nBins)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (x < nFeatures)
  {
    extern __shared__ float bins[];
    float binSize = (2.0*CUDART_PI_F)/(float)nBins;
    for (int i = 0; i < nDim; i++)
    {
      float val = features[IDX2CKernel(x,i,nDim)]*2 + CUDART_PI_F;
      for (int b = 0; b < nBins; b++)
      {
        if (val >= b*binSize && val < (b*binSize)+binSize)
        {
          bins[b] += val;
          break;
        }
      }
    }
    int maxBinIdx = 0;
    float maxBinVal = 0.0;
    for (int b = 0; b < nBins; b++)
    {
      if (bins[b] > maxBinVal)
      {
        maxBinIdx = b;
        maxBinVal = bins[b];
      }
    }
    float theta = maxBinIdx*binSize;
    for (int i = 0; i < nDim; i++)
    {
      features[IDX2CKernel(x,i,nDim)] = features[IDX2CKernel(x,i,nDim)]-theta;
    }
  }
}

// THRESHOLD INDEX
__global__ void ThresholdIdxKernel(float* pSrc, float* pDst, float threshold, int length, int* pIdx)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (x < length)
  {
    float srcVal = fabsf(pSrc[x]);
    if (srcVal < threshold)
    {
      pDst[x] = srcVal;
      pIdx[x] = x;
    } else
    {
      pDst[x] = 0;
      pIdx[x] = 0;
    }
  }
}

__global__ void GeneralizeFeatureKernel(float* features, float* outFeatures, int nFeatures, int nDim, int nFeatureWidth, int nBins)
{
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  float binSize = (CUDART_PI_F)/nBins;
  float binSum = 0;
  int localWindow = (int)sqrtf(nFeatureWidth);
  if (y < nFeatures)
  {
    int featRow = (x/localWindow)*localWindow;
    int featCol = x-featRow;
    for (int i = 0; i < localWindow; i++)
    {
      for (int j = 0; j < localWindow; j++)
      {
        int localOffset = IDX2CKernel(featRow+i,featCol+j,nFeatureWidth);
        float val = features[IDX2CKernel(y,localOffset,nDim)] + CUDART_PI_F;
        float angle = fmodf(val,CUDART_PI_F);
        for (int b = 0; b < nBins; b++)
        {
          if (angle >= b*binSize && angle < (b*binSize)+binSize)
          {
            outFeatures[IDX2CKernel(y,(x*nBins)+b,nBins*nFeatureWidth)] += val;
            binSum += val;
          }
        }
      }
    }
    for (int b = 0; b<nBins;b++)
    {
      outFeatures[IDX2CKernel(y,(x*nBins)+b,nBins*nFeatureWidth)] /= binSum;
    }
  }
}

__device__ float GetBiCubicKernelWeight(float x)
{
  if (x <= 1)
  {
    return 1.5*(x*x*x) - (2.5)*(x*x) + 1;
  } else if (x < 2)
  {
    return  (-0.5)*(x*x*x) - (5*-0.5)*(x*x) + (8*-0.5)*x + 2;
  } else
  {
    return 0;
  }
}

__device__ float ReadSubPixelKernel(float* pSrc, NppiSize oSize, float x, float y)
{
  int Xidx = (int)floorf(x);
  int Yidx = (int)floorf(y);
  float accum = 0;
  for (int v = -2; v <= 2; v++)
  {
    for (int u = -2; u <= 2; u++)
    {
      float thisV = y - v;
      float thisU = x - u;
      float k = GetBiCubicKernelWeight(sqrtf((thisV*thisV)+(thisU*thisU)));
      accum += pSrc[IDX2CKernel(Yidx+v,Xidx+u,oSize.width)]*k;
    }
  }
  return accum;
}

__global__ void MakeFeatureDescriptorKernel(float* pSrc, NppiSize oSize, float* pSubPixelX, float* pSubPixelY, int nPoints, float* pFeatures, int nFeatureWidth)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (x < nPoints)
  {
    float u = pSubPixelX[x];
    float v = pSubPixelY[x];
    float* feature = &pFeatures[x*nFeatureWidth*nFeatureWidth];
    int radius = nFeatureWidth/2;
    for (int i = 0; i < nFeatureWidth; i++)
    {
      for (int j = 0; j< nFeatureWidth; j++)
      {
          feature[IDX2CKernel(i,j,nFeatureWidth)] = ReadSubPixelKernel(pSrc,oSize,u+(i-radius),v+(j-radius));
      }
    }
  }
}

__global__ void SubPixelAlignKernel(float* pI, float* pIx, float* pIy, int* pIdx, float* pSubPixelX, float* pSubPixelY, NppiSize oSize, int nIdx)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  float I;
  float Ix;
  float Iy;

  float dX = 0;
  float dY = 0;
  if (x < nIdx)
  {
    int thisIdx = pIdx[x];
    int row = (thisIdx/oSize.width);
    int col = thisIdx - (row*oSize.width);
      int idx = IDX2CKernel(row,col,oSize.width);
      I = pI[idx];
      Ix = pIx[idx];
      Iy = pIy[idx];

      float frow = (float)row;
      float fcol = (float)col;

      if (Ix != 0)
      {
        dX = I/Ix + fcol;
      } else
      {
        dX = fcol;
      }
      if (Iy != 0)
      {
        dY = I/Iy + frow;
      } else
      {
        dY = frow;
      }
      pSubPixelY[x] = dY;
      pSubPixelX[x] = dX;
  }
}
