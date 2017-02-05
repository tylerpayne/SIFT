#include <structs/TreeNode.h>

#ifndef _BST_
#define _BST_
typedef struct BinarySearchTree BinarySearchTree;

typedef void (*addBSTNodeFunc)(BinarySearchTree*,TreeNode*);
typedef TreeNode* (*searchBSTKeyFunc)(BinarySearchTree*,float);

struct BinarySearchTree
{
  int size;
  TreeNode* root;
  addBSTNodeFunc add;
  searchBSTKeyFunc search;
  addBSTNodeFunc remove;
};

DLLEXPORT BinarySearchTree* NewBinarySearchTree();

#endif
