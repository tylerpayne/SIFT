#include <structs/Core.h>

#ifndef _TREENODE_
#define _TREENODE_

typedef struct TreeNode TreeNode;

typedef TreeNode* (*treeNodeKeyFunc)(TreeNode*,Key);
typedef int (*intTreeNode)(TreeNode*);
typedef int (*intTreeNodeTreeNode)(TreeNode*,TreeNode*);
typedef void (*voidTreeNodeTreeNode)(TreeNode*,TreeNode*);
typedef void (*voidTreeNode)(TreeNode*);


struct TreeNode
{
  Key key;
  void* value;
  TreeNode* leftChild;
  TreeNode* rightChild;
  TreeNode* parent;

  intTreeNode getHeight;
  intTreeNode getDepth;
  intTreeNode getSize;

  intTreeNode isLeftChild;
  intTreeNode isRightChild;
  intTreeNode isLeaf;
  intTreeNode isRoot;

  intTreeNode hasLeftChild;
  intTreeNode hasRightChild;

  intTreeNodeTreeNode lteTo;
  intTreeNodeTreeNode compare;

  voidTreeNodeTreeNode addRightChild;
  voidTreeNodeTreeNode addLeftChild;
  voidTreeNodeTreeNode add;
  voidTreeNodeTreeNode swap;

  treeNodeKeyFunc search;

  voidTreeNode free;
};

DLLEXPORT TreeNode* NewTreeNode(Key key);
DLLEXPORT TreeNode* NewBinaryTreeNode(Key key);
DLLEXPORT TreeNode* NewHeapNode(Key key);
#endif
