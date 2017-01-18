#include "PriorityQ.c"

typedef struct KDTree KDTree;

typedef void (*addKDTreeFunc)(KDTree*,TreeNode*);
typedef TreeNode* (*searchKDTreeFunc)(KDTree*,Matrix*,TreeNode*,int);

struct KDTree
{
  int size;
  int nPoints;
  int height;
  int kDimensions;
  TreeNode* root;
  Keypoint** data;
  searchKDTreeFunc search;
};

KDTree* NewKDTree(Keypoint** data, int nPoints, int kDimensions);
