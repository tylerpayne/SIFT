#include <stdlib.h>
#include "TreeNode.h"
#include "LinkedList.h"
#include "MatrixUtil.h"

typedef struct KDMatrixTree KDMatrixTree;

typedef TreeNode* (*searchTreeFunc)(Matrix*);
typedef void (*addTreeNodeFunc)(TreeNode*);
typedef TreeNode* (*searchFromFunc)(TreeNode*,Matrix*);
typedef TreeNode* (*getTreeNodeFunc)(int);

struct KDMatrixTree
{
  int size;
  LinkedList* nodeList;
  TreeNode* root;
  getTreeNodeFunc get;
  addTreeNodeFunc add;
  addTreeNodeFunc remove;
  searchTreeFunc search;
  searchFromFunc searchFromNode;
};

KDMatrixTree* NewKDMatrixTree(LinkedList* features);
