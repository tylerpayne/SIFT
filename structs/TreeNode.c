#include "TreeNode.h"

TreeNode* NewTreeNode(float key, void* value, int idxInTree)
{
  TreeNode* tn = (TreeNode*)malloc(sizeof(TreeNode));
  tn->key = key;
  tn->value = value;
  tn->idx = idxInTree;
  return tn;
}
