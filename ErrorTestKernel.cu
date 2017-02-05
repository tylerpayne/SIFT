__global__ void atomicAddTest(int* pI, int* pCount, int N)
{
  int x = (blockIdx.x* blockDim.x) + threadIdx.x;
  if (x < N)
  {
    int inc = 1;
    int id = atomicAdd(pCount,inc);
  }
  //pI[id] = x;
}
