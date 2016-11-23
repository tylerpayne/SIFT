#include "TreeNode.h"

TreeNode* NewTreeNode(float key)
{
  TreeNode* tn = (TreeNode*)malloc(sizeof(TreeNode));
  tn->key = key;
  tn->leftChild = NULL;
  tn->rightChild = NULL;
  tn->parent = NULL;
  tn->value = NULL;
  return tn;
}
