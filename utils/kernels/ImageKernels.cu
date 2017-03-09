#include <math_constants.h>

__device__ int IDX2CKernel(int i, int j, int td)
{
  return (i*td)+j;
}
// LocalMax
__global__ void LocalMaxKernel(float* pSrc, float* pDst, Shape oSize, int windowWidth)
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

__global__ void LocalArgMaxKernel(float* pSrc, Point2* pIdx, Shape oSize, int windowWidth)
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
      pIdx[IDX2CKernel(y,x,windowWidth)] = C2IDXKernel(maxIdx,oSize.width);
    } else
    {
      Point2 ret = {-1,-1};
      pIdx[IDX2CKernel(y,x,windowWidth)] = ret;
    }
  }
}


//Local Contrast
__global__ void LocalContrastKernel(float* pSrc, float* pDst, Shape oSize, int windowWidth)
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
  int Xidx = (int)roundf(x);
  int Yidx = (int)roundf(y);
  float accum = 0;
  for (int v = -2; v <= 2; v++)
  {
    for (int u = -2; u <= 2; u++)
    {
      float thisV = (y-Yidx) + v;
      float thisU = (x-Xidx) + u;

      float k = GetBiCubicKernelWeight(sqrtf((thisV*thisV)+(thisU*thisU)));
      int i = Yidx+v;
      int j = Xidx+u;
      i = min(i,oSize.height - 1);
      i = max(i,0);
      j = min(j,oSize.width - 1);
      j = max(j,0);
      accum += pSrc[IDX2CKernel(i,j,oSize.width)]*k;
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
    int radius = nFeatureWidth/2;
    for (int i = 0; i < nFeatureWidth; i++)
    {
      for (int j = 0; j < nFeatureWidth; j++)
      {
          pFeatures[x*nFeatureWidth*nFeatureWidth + IDX2CKernel(i,j,nFeatureWidth)] = ReadSubPixelKernel(pSrc,oSize,(u-radius)+j,(v-radius)+i);
      }
    }
  }
}

__global__ void SubPixelAlignKernel(float* pIx, float* pIy, float* pIxx, float* pIxy, float* pIyy, int* pIdx, Point2f* pSubPixel, NppiSize oSize, int nIdx)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  float Ix;
  float Iy;
  float Ixx;
  float Iyy;
  float Ixy;
  float det;

  float dX = 0;
  float dY = 0;
  if (x < nIdx)
  {
    int thisIdx = pIdx[x];
    if (thisIdx >= 0 && thisIdx < oSize.width*oSize.height)
    {
      int row = (thisIdx/oSize.width);
      int col = thisIdx - (row*oSize.width);
      int idx = IDX2CKernel(row,col,oSize.width);
      Ix = pIx[idx];
      Iy = pIy[idx];
      Ixx = pIxx[idx];
      Ixy = pIxy[idx];
      Iyy = pIyy[idx];
      det = Ixx*Iyy - Ixy*Ixy;

      float frow = (float)row;
      float fcol = (float)col;

      dX = fcol - (1.0/det)*(Ixy*Iy - Iyy*Ix);
      dY = frow - (1.0/det)*(Ixy*Ix - Ixx*Iy);

      pSubPixel[x] = {dX,dY};
    } else
    {
      pSubPixel[x] = {-1,-1};
    }
  }
}


extern __shared__ int sharedData[];
__global__ void EliminatePointsBelowThresholdKernel(float* pI, NppiSize oSize, Point2f* pSubPixel, int* pIndex, int nPoints, Point2f* keepSubPixel, int* keepPoints, int* keepCount, float threshold)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;

  int* thisKeepPoints;
  int* thisKeepCount;

  thisKeepPoints = &sharedData[x];
  thisKeepCount = &sharedData[nPoints];

  if (x < nPoints)
  {
    float thisx = pSubPixel[x].x;
    float thisy = pSubPixel[x].y;

    if (thisx >= 0 && thisy >= 0 && thisx < oSize.width && thisy < oSize.height)
    {
      float contrast = ReadSubPixelKernel(pI,oSize,thisx,thisy);
      if (contrast > threshold)
      {
        //int inc = 1;
        //unsigned int id = atomicAdd(thisKeepCount,inc);
        //if (id < nPoints)
        //{
          //thisKeepPoints[id] = x;
        //}
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < *thisKeepCount; i++)
    {
      int id = thisKeepPoints[i];
      if (id > 0)
      {
        keepSubPixel[x] = pSubPixel[thisKeepPoints[i]];
        keepPoints[x] = pIndex[thisKeepPoints[i]];
      }
    }
  }
}

__global__ void EliminateEdgePointsKernel(float* pI, NppiSize oSize, Point2f* pSubPixel, int* pIndex, int nPoints, float* pIx, float* pIy, Point2f* keepSubPixel, int* keepPoints, int* keepCount, float threshold)
{
  int x = (blockDim.x * blockIdx.x) + threadIdx.x;

  if (x < nPoints)
  {
    float thisx = pSubPixel[x].x;
    float thisy = pSubPixel[x].y;

    float Ix = ReadSubPixelKernel(pIx,oSize,thisx,thisy);
    float Iy = ReadSubPixelKernel(pIy,oSize,thisx,thisy);

    if (Ix > threshold && Iy > threshold)
    {
      float percent = fminf(Ix/Iy,Iy/Ix);
      if (percent > 0.4)
      {
        int id = atomicAdd(keepCount,1);
        keepPoints[id] = pIndex[x];
        keepSubPixel[id] = pSubPixel[x];
      }
    }

  }
}
