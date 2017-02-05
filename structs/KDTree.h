#include <structs/Core.h>
#include <utils/MatrixUtil.h>
#include <structs/Keypoint.h>
#include <structs/TreeNode.h>
#include <structs/PriorityQ.h>

#ifndef _KDTREE_
#define _KDTREE_

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

DLLEXPORT KDTree* NewKDTree(Keypoint** data, int nPoints, int kDimensions);
#endif
