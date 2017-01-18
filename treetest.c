#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "structs/Core.h"
#include "structs/BinaryTreeNode.c"
#include "utils/CUDAImageUtil.cu"
#include "structs/Keypoint.c"
#include "structs/KDTree.c"

int main(int argc, char const *argv[])
{
  switch (argc) {
    case 2:
      VERBOSITY = atoi(argv[1]);
      break;
  };
  time_t t;
  srand((unsigned) time(&t));

  MatrixUtil* matutil = GetCUDAMatrixUtil(1);

  int n, dim;
  n = 500;
  dim = 3;
  Keypoint* data = (Keypoint*)malloc(sizeof(Keypoint)*n);
  for (int i = 0; i < n; i++)
  {
    float* f = (float*)malloc(sizeof(float)*dim);
    for (int d = 0; d < dim; d++)
    {
      f[d] = (float)(rand()%50);
    }
    Matrix* m = matutil->newMatrix(f,1,dim);
    Keypoint* keypoint = NewKeypoint(10,10,NULL);
    keypoint->set(keypoint,"feature",(void*)m);
    data[i] = *keypoint;
    //free(m);
  }

  KDTree* kdtree = NewKDTree(data,n,dim);
  //printf("Tree Height: %i\n",kdtree->height);
  int k = rand()%(n-1);
  float* f = (float*)malloc(sizeof(float)*dim);
  for (int d = 0; d < dim; d++)
  {
    f[d] = (float)(rand()%50);
  }
  Matrix* mm = matutil->newMatrix(f,1,dim);
  Matrix* searching = mm;
  matutil->pprint(matutil,searching,"Searching For");
  TreeNode* nextBest = (TreeNode*)malloc(sizeof(TreeNode));

  TreeNode* found = kdtree->search(kdtree,searching,nextBest,1);

  return 0;
}
